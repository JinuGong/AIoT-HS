# Jetson Orin Nano Super Roofline 실습

## 1. 실습 목표

본 실습에서는 **Jetson Orin Nano Super**의 GPU 성능 특성을 Roofline Model 관점에서 직접 측정하고 시각화한다.

주요 목표는 다음과 같다.

- Jetson의 최대 성능 모드(`MAXN_SUPER`) 설정
- GPU FP32 compute throughput 측정
- DRAM memory bandwidth 측정
- Arithmetic Intensity(AI)를 변화시키는 CUDA kernel 작성
- Memory-bound / Compute-bound 전환 관찰
- Theoretical Roofline과 Empirical Roofline 비교
- 최종 Roofline plot 생성

Roofline Model은 다음 식으로 표현된다.

\[
P(AI)=\min(P_{\mathrm{peak}}, BW_{\mathrm{mem}}\times AI)
\]

여기서

- \(P\): 실제 성능 [FLOP/s]
- \(P_{\mathrm{peak}}\): 최대 compute throughput [FLOP/s]
- \(BW_{\mathrm{mem}}\): memory bandwidth [Byte/s]
- \(AI\): arithmetic intensity [FLOP/Byte]

이다.

Arithmetic Intensity는

\[
AI=\frac{\text{FLOPs}}{\text{Memory Traffic [Byte]}}
\]

로 정의한다.

---

## 2. 실습 환경

이번 실험에서 사용한 장치는 다음과 같다.

| 항목 | 값 |
|---|---|
| Device | NVIDIA Jetson Orin Nano Super Developer Kit |
| Architecture | NVIDIA Ampere |
| Compute Capability | 8.7 |
| GPU SM count | 8 |
| RAM | 8 GB unified memory |
| CUDA | 12.6 |
| Jetson Linux | R36.4.7 |
| Kernel | Linux 5.15.148-tegra |
| Power Mode | MAXN_SUPER |
| GPU Clock | 1.020 GHz |
| CPU Clock | 1.728 GHz |
| EMC Clock | 3.199 GHz |

장치 정보 확인:

```bash
uname -a
cat /etc/nv_tegra_release
nvcc --version
nvidia-smi
cat /proc/device-tree/model
```

전력 모드 및 클럭 확인:

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

---

## 3. MAXN_SUPER 모드 설정

Roofline 측정에서는 DVFS에 의해 GPU 및 memory clock이 변하면 측정값이 흔들릴 수 있으므로 최대 성능 모드로 고정한다.

Power mode 확인:

```bash
sudo nvpmodel -q
```

이번 장치에서 MAXN_SUPER는 mode ID `2`였다.

```bash
sudo nvpmodel -m 2
sudo jetson_clocks
```

설정 후 확인:

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

실제 확인 결과:

```text
NV Power Mode: MAXN_SUPER

CPU MaxFreq = 1728000
GPU MaxFreq = 1020000000
EMC CurrentFreq = 3199000000
```

즉 실험 동안

- CPU: 1.728 GHz
- GPU: 1.020 GHz
- EMC: 3.199 GHz

상태로 고정하였다.

---

## 4. Roofline 실험 디렉터리 생성

```bash
mkdir -p ~/roofline_lab
cd ~/roofline_lab
```

---

## 5. 초기 Memory / FP32 Compute Benchmark

처음에는 두 개의 간단한 microbenchmark를 이용하여

1. memory bandwidth
2. FP32 compute throughput

을 측정하였다.

초기 측정 결과는 다음과 같았다.

```text
GPU: Orin
SM count: 8
Compute capability: 8.7

=== Memory Benchmark ===
Measured bandwidth: 60.81 GB/s

=== FP32 Compute Benchmark ===
Measured FP32 throughput: 1290.36 GFLOP/s

=== Roofline Parameters ===
Memory roof : 60.81 GB/s
Compute roof: 1290.36 GFLOP/s
Ridge point : 21.22 FLOP/Byte
```

초기에는 이 값을 empirical roof로 사용하였다.

\[
P_{\mathrm{empirical}}
=
\min(1290.36,\;60.81\times AI)
\]

그러나 이후 AI sweep 실험에서 이 값보다 높은 성능이 관찰되었다.

따라서 위 값들은 **Jetson GPU 전체의 실제 ceiling이 아니라 특정 microbenchmark가 달성한 성능**임을 확인하였다.

이 점은 Roofline 실험에서 중요하다.

> 하나의 kernel에서 측정한 throughput을 곧바로 hardware roof로 간주하면 안 된다.

---

## 6. Arithmetic Intensity Sweep 설계

실제 Roofline behavior를 확인하기 위해 Arithmetic Intensity를 인위적으로 증가시키는 CUDA kernel을 사용하였다.

핵심 구조:

```cpp
float x = input[i];

for (int k = 0; k < K; ++k) {
    x = fmaf(x, a, b);
}

output[i] = x;
```

각 element에 대해 memory traffic은

- FP32 input read: 4 Byte
- FP32 output write: 4 Byte

이므로 총

\[
8\text{ Byte}
\]

이다.

FMA 한 번은

\[
a\times b+c
\]

형태이므로 2 FLOPs로 계산한다.

따라서 반복 횟수가 \(K\)일 때

\[
\text{FLOPs}=2K
\]

이고 Arithmetic Intensity는

\[
AI=\frac{2K}{8}
=\frac{K}{4}
\]

이다.

따라서 다음과 같이 AI를 변화시켰다.

| K | AI [FLOP/Byte] |
|---:|---:|
| 1 | 0.25 |
| 2 | 0.50 |
| 4 | 1 |
| 8 | 2 |
| 16 | 4 |
| 32 | 8 |
| 64 | 16 |
| 128 | 32 |
| 256 | 64 |
| 512 | 128 |
| 1024 | 256 |
| 2048 | 512 |

---

## 7. AI Sweep CUDA 코드

`roofline_sweep.cu`

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <fstream>

#define CHECK_CUDA(call)                                              \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)        \
                  << " at line " << __LINE__ << std::endl;            \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

template<int K>
__global__ void ai_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    float x = in[i];

    const float a = 1.000001f;
    const float b = 0.000001f;

    #pragma unroll
    for (int k = 0; k < K; ++k) {
        x = fmaf(x, a, b);
    }

    out[i] = x;
}

template<int K>
void benchmark(
    const float* d_in,
    float* d_out,
    size_t n,
    int blocks,
    int threads,
    int repeat,
    std::ofstream& csv)
{
    ai_kernel<K><<<blocks, threads>>>(d_in, d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int r = 0; r < repeat; ++r) {
        ai_kernel<K><<<blocks, threads>>>(d_in, d_out, n);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double bytes_per_element = 8.0;
    double flops_per_element = 2.0 * static_cast<double>(K);

    double arithmetic_intensity =
        flops_per_element / bytes_per_element;

    double total_flops =
        static_cast<double>(n)
        * flops_per_element
        * repeat;

    double seconds = ms / 1000.0;

    double gflops =
        total_flops / seconds / 1e9;

    double total_bytes =
        static_cast<double>(n)
        * bytes_per_element
        * repeat;

    double bandwidth =
        total_bytes / seconds / 1e9;

    std::cout
        << std::setw(6) << K
        << "   "
        << std::setw(10)
        << std::fixed
        << std::setprecision(2)
        << arithmetic_intensity
        << "   "
        << std::setw(12)
        << gflops
        << " GFLOP/s   "
        << std::setw(10)
        << bandwidth
        << " GB/s"
        << std::endl;

    csv
        << K << ","
        << arithmetic_intensity << ","
        << gflops << ","
        << bandwidth << ","
        << ms
        << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main()
{
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    std::cout
        << "GPU: "
        << prop.name
        << "\n";

    std::cout
        << "SM count: "
        << prop.multiProcessorCount
        << "\n\n";

    const size_t N =
        32ULL * 1024 * 1024;

    const size_t bytes =
        N * sizeof(float);

    float* d_in;
    float* d_out;

    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    CHECK_CUDA(cudaMemset(d_in, 0, bytes));
    CHECK_CUDA(cudaMemset(d_out, 0, bytes));

    const int threads = 256;
    const int blocks =
        (N + threads - 1) / threads;

    const int repeat = 10;

    std::ofstream csv(
        "roofline_points.csv"
    );

    csv
        << "K,"
        << "AI_FLOP_per_Byte,"
        << "GFLOPs,"
        << "Effective_BW_GBs,"
        << "Elapsed_ms\n";

    std::cout
        << "K"
        << "      AI"
        << "       Performance"
        << "       Effective BW"
        << "\n";

    std::cout
        << "---------------------------------------------------------"
        << "\n";

    benchmark<1>(d_in, d_out, N, blocks, threads, repeat, csv);
    benchmark<2>(d_in, d_out, N, blocks, threads, repeat, csv);
    benchmark<4>(d_in, d_out, N, blocks, threads, repeat, csv);
    benchmark<8>(d_in, d_out, N, blocks, threads, repeat, csv);
    benchmark<16>(d_in, d_out, N, blocks, threads, repeat, csv);
    benchmark<32>(d_in, d_out, N, blocks, threads, repeat, csv);
    benchmark<64>(d_in, d_out, N, blocks, threads, repeat, csv);
    benchmark<128>(d_in, d_out, N, blocks, threads, repeat, csv);
    benchmark<256>(d_in, d_out, N, blocks, threads, repeat, csv);
    benchmark<512>(d_in, d_out, N, blocks, threads, repeat, csv);
    benchmark<1024>(d_in, d_out, N, blocks, threads, repeat, csv);
    benchmark<2048>(d_in, d_out, N, blocks, threads, repeat, csv);

    csv.close();

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    std::cout
        << "\nSaved: roofline_points.csv\n";

    return 0;
}
```

컴파일:

```bash
nvcc -O3 -arch=sm_87 roofline_sweep.cu -o roofline_sweep
```

실행:

```bash
./roofline_sweep
```

---

## 8. 실제 AI Sweep 결과

실험 결과:

| K | AI [FLOP/Byte] | Performance [GFLOP/s] | Effective BW [GB/s] |
|---:|---:|---:|---:|
| 1 | 0.25 | 18.56 | 74.26 |
| 2 | 0.50 | 37.07 | 74.14 |
| 4 | 1.00 | 74.15 | 74.15 |
| 8 | 2.00 | 146.42 | 73.21 |
| 16 | 4.00 | 286.78 | 71.70 |
| 32 | 8.00 | 548.43 | 68.55 |
| 64 | 16.00 | 939.87 | 58.74 |
| 128 | 32.00 | 1297.61 | 40.55 |
| 256 | 64.00 | 1629.96 | 25.47 |
| 512 | 128.00 | 1845.68 | 14.42 |
| 1024 | 256.00 | 1962.63 | 7.67 |
| 2048 | 512.00 | 2018.83 | 3.94 |

---

## 9. Memory-bound 영역

낮은 AI에서는 performance가 거의

\[
P \approx BW\times AI
\]

형태를 따른다.

예를 들어

\[
AI=1
\]

일 때

\[
P=74.15\ \text{GFLOP/s}
\]

이고 effective bandwidth는

\[
74.15\ \text{GB/s}
\]

이다.

또한

\[
AI=2
\]

에서는

\[
P=146.42\ \text{GFLOP/s}
\]

이므로 거의

\[
74\times2
\]

의 관계가 성립한다.

즉 낮은 AI 영역에서는 성능이 compute throughput이 아니라 **memory bandwidth에 의해 제한**된다.

실험에서 낮은 AI 구간의 sustained bandwidth는 약

\[
\boxed{74.2\ \text{GB/s}}
\]

로 관찰되었다.

---

## 10. Compute-bound 영역

AI가 증가할수록 performance는 계속 증가하지만 어느 순간 증가율이 크게 감소한다.

```text
K=256   → 1629.96 GFLOP/s
K=512   → 1845.68 GFLOP/s
K=1024  → 1962.63 GFLOP/s
K=2048  → 2018.83 GFLOP/s
```

즉 약

\[
\boxed{2.0\ \text{TFLOP/s}}
\]

근처에서 saturation이 발생한다.

따라서 이번 실험의 empirical FP32 compute roof는

\[
\boxed{2018.83\ \text{GFLOP/s}}
\]

정도로 볼 수 있다.

---

## 11. Theoretical Roofline

Jetson Orin Nano Super의 이번 실험 조건에서 theoretical parameter를 다음과 같이 사용하였다.

### FP32 Compute

GPU:

- SM: 8
- FP32 cores per SM: 128
- GPU clock: 1.02 GHz
- FMA: 2 FLOPs

따라서

\[
P_{\mathrm{theory}}
=
8\times128\times1.02\times2
\]

\[
\approx2089\ \text{GFLOP/s}
\]

즉

\[
\boxed{2.09\ \text{TFLOP/s}}
\]

이다.

### Memory Bandwidth

Theoretical DRAM bandwidth:

\[
\boxed{102.4\ \text{GB/s}}
\]

따라서 theoretical Roofline은

\[
P_{\mathrm{theory}}(AI)
=
\min
\left(
2089,\;
102.4\times AI
\right)
\]

이다.

---

## 12. Empirical Roofline

이번 AI sweep 결과를 기준으로

\[
BW_{\mathrm{empirical}}
\approx74.2\ \text{GB/s}
\]

\[
P_{\mathrm{empirical}}
\approx2018.83\ \text{GFLOP/s}
\]

를 사용하였다.

따라서

\[
P_{\mathrm{empirical}}(AI)
=
\min
\left(
2018.83,\;
74.2\times AI
\right)
\]

이다.

---

## 13. Ridge Point

Ridge point는 memory-bound 영역과 compute-bound 영역이 만나는 지점이다.

\[
AI_{\mathrm{ridge}}
=
\frac{P_{\mathrm{peak}}}{BW}
\]

### Theoretical Ridge

\[
AI_{\mathrm{ridge,theory}}
=
\frac{2089}{102.4}
\approx20.4
\]

따라서

\[
\boxed{20.4\ \text{FLOP/Byte}}
\]

이다.

### Empirical Ridge

\[
AI_{\mathrm{ridge,empirical}}
=
\frac{2018.83}{74.2}
\approx27.2
\]

따라서

\[
\boxed{27.2\ \text{FLOP/Byte}}
\]

이다.

실제 측정 결과도

```text
AI = 16   → memory 영향이 아직 큼
AI = 32   → transition 영역
AI = 64   → compute-bound 영역 진입
AI ≥ 128  → compute ceiling에 접근
```

하는 모습을 보였다.

---

## 14. Effective Bandwidth 감소의 의미

실험에서 AI가 증가할수록 다음과 같이 Effective BW가 감소하였다.

```text
AI = 1     → 74.15 GB/s
AI = 16    → 58.74 GB/s
AI = 64    → 25.47 GB/s
AI = 256   → 7.67 GB/s
AI = 512   → 3.94 GB/s
```

이것은 DRAM 자체가 느려진 것을 의미하지 않는다.

본 실험에서 Effective BW는

\[
BW_{\mathrm{effective}}
=
\frac{\text{memory traffic}}
{\text{execution time}}
\]

으로 계산한다.

AI가 증가하면 같은 memory traffic에 대해 더 많은 computation을 수행하게 된다.

따라서 실행 시간 중 compute가 차지하는 비율이 증가하면서 단위 시간당 memory traffic은 감소한다.

즉 compute-bound 영역에서 Effective BW가 감소하는 것은 정상적인 현상이다.

---

## 15. Roofline Plot 코드

`plot_roofline.py`

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Hardware / measured parameters
# ============================================================

THEORY_BW = 102.4
THEORY_FP32 = 2089.0

MEASURED_BW = 74.2
MEASURED_FP32 = 2018.83


# ============================================================
# 2. Load measured kernel points
# ============================================================

df = pd.read_csv("roofline_points.csv")


# ============================================================
# 3. Ridge points
# ============================================================

ridge_theory = THEORY_FP32 / THEORY_BW
ridge_empirical = MEASURED_FP32 / MEASURED_BW

print(
    f"Theoretical ridge point : "
    f"{ridge_theory:.2f} FLOP/Byte"
)

print(
    f"Empirical ridge point   : "
    f"{ridge_empirical:.2f} FLOP/Byte"
)


# ============================================================
# 4. Roofline curves
# ============================================================

ai = np.logspace(
    -2,
    3,
    1000
)

theoretical_roof = np.minimum(
    THEORY_BW * ai,
    THEORY_FP32
)

empirical_roof = np.minimum(
    MEASURED_BW * ai,
    MEASURED_FP32
)


# ============================================================
# 5. Plot
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.loglog(
    ai,
    theoretical_roof,
    "--",
    linewidth=2,
    label=(
        f"Theoretical Roofline "
        f"({THEORY_BW:.1f} GB/s, "
        f"{THEORY_FP32/1000:.2f} TFLOP/s)"
    )
)

plt.loglog(
    ai,
    empirical_roof,
    linewidth=2,
    label=(
        f"Empirical Roofline "
        f"({MEASURED_BW:.1f} GB/s, "
        f"{MEASURED_FP32/1000:.2f} TFLOP/s)"
    )
)

plt.scatter(
    df["AI_FLOP_per_Byte"],
    df["GFLOPs"],
    s=65,
    zorder=5,
    label="Measured Kernels"
)


for _, row in df.iterrows():

    x = row["AI_FLOP_per_Byte"]
    y = row["GFLOPs"]
    k = int(row["K"])

    plt.annotate(
        f"K={k}",
        (x, y),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8
    )


# ============================================================
# 6. Ridge points
# ============================================================

plt.axvline(
    ridge_theory,
    linestyle=":",
    linewidth=1.5
)

plt.axvline(
    ridge_empirical,
    linestyle=":",
    linewidth=1.5
)

plt.text(
    ridge_theory * 0.93,
    2.0,
    f"Theoretical Ridge\n"
    f"{ridge_theory:.1f} FLOP/Byte",
    rotation=90,
    verticalalignment="bottom",
    horizontalalignment="right",
    fontsize=9
)

plt.text(
    ridge_empirical * 1.07,
    2.0,
    f"Empirical Ridge\n"
    f"{ridge_empirical:.1f} FLOP/Byte",
    rotation=90,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=9
)


# ============================================================
# 7. Region labels
# ============================================================

plt.text(
    0.06,
    500,
    "Memory-bound",
    fontsize=11
)

plt.text(
    120,
    700,
    "Compute-bound",
    fontsize=11
)


# ============================================================
# 8. Axis / title
# ============================================================

plt.xlabel(
    "Arithmetic Intensity [FLOP/Byte]",
    fontsize=12
)

plt.ylabel(
    "Performance [GFLOP/s]",
    fontsize=12
)

plt.title(
    "Jetson Orin Nano Super\n"
    "Theoretical vs. Empirical FP32 Roofline",
    fontsize=14
)

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.legend(
    loc="upper left"
)

plt.tight_layout()


# ============================================================
# 9. Save
# ============================================================

plt.savefig(
    "roofline_final.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
```

필요 패키지:

```bash
python3 -m pip install numpy pandas matplotlib
```

실행:

```bash
python3 plot_roofline.py
```

결과:

```text
roofline_final.png
```

---

## 16. 최종 결과 해석

이번 Jetson Orin Nano Super 실험에서 다음과 같은 결과를 얻었다.

### Theoretical

\[
BW_{\mathrm{theory}}
=
102.4\ \text{GB/s}
\]

\[
P_{\mathrm{theory}}
=
2.09\ \text{TFLOP/s}
\]

\[
AI_{\mathrm{ridge,theory}}
=
20.4\ \text{FLOP/Byte}
\]

### Empirical

\[
BW_{\mathrm{empirical}}
\approx
74.2\ \text{GB/s}
\]

\[
P_{\mathrm{empirical}}
\approx
2.02\ \text{TFLOP/s}
\]

\[
AI_{\mathrm{ridge,empirical}}
\approx
27.2\ \text{FLOP/Byte}
\]

특히 낮은 AI에서는 측정점들이 empirical memory roof를 거의 그대로 따라가며, 높은 AI에서는 약 2 TFLOP/s 부근으로 수렴하였다.

이는 Roofline Model이 설명하는

```text
낮은 Arithmetic Intensity
        ↓
Memory-bound
        ↓
AI 증가
        ↓
Ridge Point
        ↓
Compute-bound
```

전환을 실제 Jetson GPU에서 직접 확인한 결과이다.

---
