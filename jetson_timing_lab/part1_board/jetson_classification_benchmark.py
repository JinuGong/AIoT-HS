#!/usr/bin/env python3
"""Compare Jetson GPU-only CNN inference latency without a dataset or torchvision."""
import argparse
import glob
import json
import platform
import subprocess
import statistics
import time
from pathlib import Path

import torch
from torch import nn


def read_power_mode():
    """Best-effort audit record; benchmark still works without nvpmodel permissions."""
    try:
        result = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=5, check=False
        )
        return (result.stdout or result.stderr).strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def read_sysfs_snapshot():
    """Best-effort: thermal zone temperatures (C) and devfreq clocks (MHz). No sudo needed."""
    snap = {"temps_c": {}, "devfreq_mhz": {}}
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            name = Path(zone, "type").read_text().strip()
            snap["temps_c"][name] = round(int(Path(zone, "temp").read_text()) / 1000, 1)
        except (OSError, ValueError, TypeError, UnicodeError):
            pass
    for dev in sorted(glob.glob("/sys/class/devfreq/*")):
        try:
            name = Path(dev).name
            snap["devfreq_mhz"][name] = round(int(Path(dev, "cur_freq").read_text()) / 1e6, 1)
        except (OSError, ValueError, TypeError, UnicodeError):
            pass
    return snap


def make_model():
    # A fixed 224x224 image classification network; no downloads needed.
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10),
    )


def latency(model, batch_size, warmup, repeats):
    # Same tensor size and RNG seed on all boards. Input already resides in GPU memory.
    generator = torch.Generator(device="cpu").manual_seed(2026 + batch_size)
    inp = torch.randn(batch_size, 3, 224, 224, generator=generator).cuda()
    with torch.inference_mode():
        for _ in range(warmup):
            model(inp)
        torch.cuda.synchronize()

        times = []
        for _ in range(3):  # independent trials for stability
            events = []
            for _ in range(repeats):
                begin = torch.cuda.Event(enable_timing=True)
                finish = torch.cuda.Event(enable_timing=True)
                begin.record()
                model(inp)
                finish.record()
                events.append((begin, finish))
            torch.cuda.synchronize()
            elapsed = [s.elapsed_time(e) for s, e in events]
            elapsed.sort()
            times.append({
                "median_ms": statistics.median(elapsed),
                "p95_ms": elapsed[int(0.95 * (len(elapsed) - 1))],
            })
        # Host-side wall clock per forward (synchronize each call): what an application waits for,
        # including Python/kernel-launch overhead. Reported separately from the CUDA-event time.
        wall = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(inp)
            torch.cuda.synchronize()
            wall.append((time.perf_counter() - t0) * 1000)
    median_ms = statistics.median(t["median_ms"] for t in times)
    return {
        "batch_size": batch_size,
        "median_ms": round(median_ms, 4),
        "p95_ms": round(statistics.median(t["p95_ms"] for t in times), 4),
        "images_per_second": round(batch_size * 1000 / median_ms, 2),
        "trial_median_ms": [round(t["median_ms"], 4) for t in times],
        "wall_median_ms": round(statistics.median(wall), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmark_result.json")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 10:
        parser.error("warmup must be >=0 and repeats must be >=10")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: please run on the Jetson PyTorch CUDA environment")

    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = True
    model = make_model().cuda().eval()
    data = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0),
        "power_mode_report": read_power_mode(),
        "cudnn_benchmark": True,
        "sysfs_before": read_sysfs_snapshot(),
        "precision": "FP32 (CUDA/cuDNN defaults)",
        "input_shape": "(batch_size, 3, 224, 224)",
        "model": "fixed-four-conv-CNN (untrained, benchmark only)",
        "warmup": args.warmup,
        "repeats_per_trial": args.repeats,
        "results": [],
    }
    for batch in (1, 16):
        result = latency(model, batch, args.warmup, args.repeats)
        data["results"].append(result)
        print(f"B={batch:2d} | median={result['median_ms']:.4f} ms/batch "
              f"| p95={result['p95_ms']:.4f} ms/batch "
              f"| throughput={result['images_per_second']:.2f} img/s "
              f"| wall={result['wall_median_ms']:.4f} ms")
    data["sysfs_after"] = read_sysfs_snapshot()
    Path(args.out).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
