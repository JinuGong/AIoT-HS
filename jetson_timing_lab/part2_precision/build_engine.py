#!/usr/bin/env python3
"""2단계: mnist.onnx → TensorRT engine. 세 precision 모두 같은 함수로 만든다 (다른 것은 builder flag 한 줄).

사용:
    python3 build_engine.py fp32        # → mnist_fp32.engine  (TF32 끔: 진짜 FP32 기준)
    python3 build_engine.py fp16        # → mnist_fp16.engine
    python3 build_engine.py int8        # → mnist_int8.engine  (calib_data.npy로 PTQ calibration)

INT8만 calibrator가 필요한 이유: weight의 범위는 파일에 있지만, activation(중간 출력)의 범위는
실제 입력을 넣어 봐야 알 수 있다. 아래 Calibrator는 calib_data.npy를 batch 단위로 GPU에 올려 줄 뿐이다.
"""
import argparse
import os

import numpy as np
import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)
SHAPES = dict(min=(1, 1, 28, 28), opt=(32, 1, 28, 28), max=(128, 1, 28, 28))


class Calibrator(trt.IInt8EntropyCalibrator2):
    """calib_data.npy를 batch씩 GPU 메모리(torch 텐서)에 복사해 TensorRT에 주소만 넘긴다. pycuda 불필요."""
    def __init__(self, npy="calib_data.npy", batch=32, cache="mnist_calib.cache"):
        super().__init__()
        self.data = torch.from_numpy(np.load(npy).astype(np.float32))
        self.buf = torch.empty((batch, 1, 28, 28), dtype=torch.float32, device="cuda")
        self.batch, self.cache, self.i = batch, cache, 0

    def get_batch_size(self):
        return self.batch

    def get_batch(self, names):
        if self.i + self.batch > len(self.data):
            return None                                   # 데이터 끝 → calibration 종료
        self.buf.copy_(self.data[self.i:self.i + self.batch]); torch.cuda.synchronize()
        self.i += self.batch
        return [self.buf.data_ptr()]

    def read_calibration_cache(self):
        return open(self.cache, "rb").read() if os.path.exists(self.cache) else None

    def write_calibration_cache(self, cache):
        open(self.cache, "wb").write(cache)


def build(precision, onnx_path="mnist.onnx"):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)                   # TensorRT 10: 항상 explicit batch
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(open(onnx_path, "rb").read()):
        raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape("input", **SHAPES)
    config.add_optimization_profile(profile)

    if precision == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)           # Ampere(Orin)는 기본으로 TF32를 허용 → 끄고 순수 FP32 기준
    elif precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)             # INT8이 안 되는 layer는 FP32 대신 FP16으로 (TensorRT 일반 권장)
        config.int8_calibrator = Calibrator()
        config.set_calibration_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise SystemExit("engine 생성 실패 (위 로그 확인)")
    out = f"mnist_{precision}.engine"
    open(out, "wb").write(engine)
    print(f"Saved: {out}  ({os.path.getsize(out) / 1024:.0f} KB)  TensorRT {trt.__version__}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("precision", choices=["fp32", "fp16", "int8"])
    build(ap.parse_args().precision)
