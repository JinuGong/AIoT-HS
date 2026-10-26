#!/usr/bin/env python3
"""3단계: 정확도 + 시간을 같은 스크립트, 같은 방법으로 측정한다 (보드 기준 실험과 같은 CUDA event 방식).

사용:
    python3 eval_engine.py mnist_fp32.engine mnist_fp16.engine mnist_int8.engine --torch mnist_fp32.pth --out precision.json

- 정확도: test_data.npy 10,000장 전체
- 시간: 입력을 GPU에 미리 둔 상태에서 B=1, B=32 각각 warm-up 100회 + 3 trial × 200회, median / p95 (ms/batch)
- --torch: 같은 가중치를 PyTorch(FP32)로 실행한 값도 함께 → "런타임(PyTorch vs TensorRT)" 효과 비교
"""
import argparse
import json
import os
import statistics

import numpy as np
import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTModel:
    """engine 하나를 torch 텐서 버퍼로 실행한다 (TensorRT 10 API, pycuda 불필요)."""
    def __init__(self, path):
        self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream()          # torch와 같은 stream → 버퍼 복사와 순서가 보장됨

    def __call__(self, x):                                 # x: (B,1,28,28) float32 CUDA tensor
        self.ctx.set_input_shape("input", tuple(x.shape))
        out = torch.empty(tuple(self.ctx.get_tensor_shape("output")), dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address("input", x.data_ptr())
        self.ctx.set_tensor_address("output", out.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        return out


def accuracy(run, x, y, batch=128):
    correct = 0
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].contiguous()
        out = run(xb)
        torch.cuda.synchronize()
        correct += (out.argmax(1) == y[i:i + batch]).sum().item()
    return correct / len(x) * 100


def timing(run, stream, B, warmup=100, repeats=200):
    x = torch.randn(B, 1, 28, 28, generator=torch.Generator().manual_seed(2026 + B)).cuda().contiguous()
    torch.cuda.synchronize()
    for _ in range(warmup):
        run(x)
    torch.cuda.synchronize()
    meds, p95s = [], []
    for _ in range(3):
        ev = []
        for _ in range(repeats):
            b, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            b.record(stream); run(x); e.record(stream); ev.append((b, e))
        torch.cuda.synchronize()
        t = sorted(b.elapsed_time(e) for b, e in ev)
        meds.append(statistics.median(t)); p95s.append(t[int(0.95 * (len(t) - 1))])
    med = statistics.median(meds)
    return {"batch": B, "median_ms": round(med, 4), "p95_ms": round(statistics.median(p95s), 4),
            "images_per_s": round(B * 1000 / med, 1), "trial_median_ms": [round(m, 4) for m in meds]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="*")
    ap.add_argument("--torch", help="PyTorch FP32 가중치(.pth)도 같은 방법으로 측정")
    ap.add_argument("--out", default="precision.json")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")

    x = torch.from_numpy(np.load("test_data.npy").astype(np.float32)).cuda()
    y = torch.from_numpy(np.load("test_labels.npy")).cuda()
    rows = []

    if a.torch:
        from train_export import SmallCNN
        m = SmallCNN().cuda().eval(); m.load_state_dict(torch.load(a.torch, map_location="cuda"))
        s = torch.cuda.current_stream()
        def run_t(xb):
            with torch.inference_mode():
                return m(xb)
        rows.append({"name": "PyTorch FP32", "file_KB": round(os.path.getsize(a.torch) / 1024),
                     "accuracy_%": round(accuracy(run_t, x, y), 2), "timing": [timing(run_t, s, B) for B in (1, 32)]})

    for path in a.engines:
        m = TRTModel(path)
        rows.append({"name": "TensorRT " + path.split("_")[-1].split(".")[0].upper(), "file_KB": round(os.path.getsize(path) / 1024),
                     "accuracy_%": round(accuracy(m, x, y), 2), "timing": [timing(m, m.stream, B) for B in (1, 32)]})

    print(f"\n{'model':16s} {'size KB':>8s} {'acc %':>7s} {'B=1 ms':>9s} {'B=1 p95':>9s} {'B=32 ms':>9s} {'B=32 img/s':>11s}")
    for r in rows:
        t1, t32 = r["timing"]
        print(f"{r['name']:16s} {r['file_KB']:8d} {r['accuracy_%']:7.2f} {t1['median_ms']:9.4f} {t1['p95_ms']:9.4f} "
              f"{t32['median_ms']:9.4f} {t32['images_per_s']:11.0f}")
    meta = {"tensorrt": trt.__version__, "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    json.dump({"meta": meta, "results": rows}, open(a.out, "w"), indent=2, ensure_ascii=False)
    print("Saved:", a.out)


if __name__ == "__main__":
    main()
