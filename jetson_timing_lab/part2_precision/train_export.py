#!/usr/bin/env python3
"""1단계: MNIST CNN을 FP32로 학습하고 ONNX로 내보낸다.

출력: mnist_fp32.pth, mnist.onnx, calib_data.npy (학습 이미지 1024장), test_data.npy, test_labels.npy
사용: python3 train_export.py            (--epochs 3)
"""
import argparse
import gzip
import inspect
import os
import struct
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


class SmallCNN(nn.Module):
    """Conv(1→16) → Pool → Conv(16→32) → Pool → FC(1568→128) → FC(128→10)"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def load_mnist(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    for f in FILES:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            print("Downloading:", f)
            urllib.request.urlretrieve(MNIST_URL + f, path)

    def images(f):
        with gzip.open(os.path.join(data_dir, f), "rb") as fh:
            _, n, r, c = struct.unpack(">IIII", fh.read(16))
            return (np.frombuffer(fh.read(), np.uint8).reshape(n, 1, r, c).astype(np.float32) / 255.0)

    def labels(f):
        with gzip.open(os.path.join(data_dir, f), "rb") as fh:
            struct.unpack(">II", fh.read(8))
            return np.frombuffer(fh.read(), np.uint8).astype(np.int64)

    return images(FILES[0]), labels(FILES[1]), images(FILES[2]), labels(FILES[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--calib", type=int, default=1024, help="INT8 calibration에 쓸 학습 이미지 수")
    a = ap.parse_args()

    torch.manual_seed(2026); np.random.seed(2026)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

    tx, ty, vx, vy = load_mnist()
    print("Train:", tx.shape, "Test:", vx.shape)
    train = DataLoader(TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty)), batch_size=128, shuffle=True)
    test = DataLoader(TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy)), batch_size=256)

    model = SmallCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(a.epochs):
        model.train(); total = 0.0
        for x, y in train:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
            total += loss.item()
        print(f"Epoch {ep + 1}/{a.epochs}  loss={total / len(train):.4f}")

    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in test:
            correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
    print(f"PyTorch FP32 accuracy: {correct / len(vy) * 100:.2f}%")

    torch.save(model.state_dict(), "mnist_fp32.pth")
    np.save("calib_data.npy", tx[:a.calib])
    np.save("test_data.npy", vx); np.save("test_labels.npy", vy)

    # ONNX: batch 차원만 가변. 새 exporter(dynamo)가 기본인 PyTorch 버전에서도 같은 결과가 나오도록 고전 exporter 사용
    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(model.cpu(), torch.randn(1, 1, 28, 28), "mnist.onnx",
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, opset_version=17, **kw)
    import onnx
    onnx.checker.check_model(onnx.load("mnist.onnx"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Saved: mnist_fp32.pth, mnist.onnx (ONNX OK) | parameters = {n_params:,} (FP32 {n_params * 4 / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
