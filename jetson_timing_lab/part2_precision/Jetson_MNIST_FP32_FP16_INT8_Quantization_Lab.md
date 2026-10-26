# Jetson Orin Nano에서 FP32 / FP16 / INT8 Quantization 실험

## 1. 실험 목적

Jetson Orin Nano에서 간단한 CNN을 직접 학습한 뒤, **같은 모델**을 TensorRT로 FP32 · FP16 · INT8 engine으로 변환하고 다음을 비교한다.

- 분류 정확도 (test set 10,000장)
- TensorRT engine 크기
- 추론 시간 (median · p95) — **보드 기준 실험과 같은 CUDA event 방식**
- 처리량 (images/s)

```text
MNIST CNN 학습 (PyTorch, FP32)
  ↓  train_export.py
mnist.onnx  (+ calib_data.npy, test_data.npy)
  ↓  build_engine.py fp32 | fp16 | int8      ← 세 precision 모두 같은 스크립트
mnist_fp32.engine · mnist_fp16.engine · mnist_int8.engine
  ↓  eval_engine.py
정확도 + 시간 + 크기 한 표 (precision.json)
  ↓  (확인용) trtexec
```

> 이 실험은 **보드 기준 실험(jetson_classification_benchmark.py)** 다음에 한다. 보드가 공정한지 먼저 확인해야 팀끼리 precision 결과를 비교할 수 있다.

---

## 2. 실험 환경

| 항목 | 값 (예시 보드에서 확인) |
|---|---|
| Device | Jetson Orin Nano (Super), GPU compute capability 8.7 (Ampere), 8 SM |
| 전력 모드 | MAXN SUPER + `jetson_clocks` (보드 기준 실험과 동일) |
| TensorRT | 10.3.0 (JetPack 6.x 포함) |
| Python | 3.10 (JetPack 시스템 Python) |

```bash
python3 -c "import tensorrt as trt; print(trt.__version__)"
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3. 가상환경

Jetson의 TensorRT · PyTorch는 JetPack용 빌드이므로 **시스템 패키지를 그대로 보이게** 가상환경을 만든다.

```bash
mkdir -p ~/quantization_lab && cd ~/quantization_lab
/usr/bin/python3 -m venv --system-site-packages quant_env
source quant_env/bin/activate
pip install onnx "numpy==1.26.4"     # NumPy 2.x는 JetPack용 PyTorch와 binary 호환 문제가 날 수 있음
```

- **pycuda는 설치하지 않는다.** GPU 버퍼는 PyTorch 텐서로 대신한다.
- `pip install torch`는 하지 않는다 (JetPack용 torch를 덮어쓰면 CUDA가 사라진다).

실습 파일 3개를 이 폴더에 둔다: `train_export.py`, `build_engine.py`, `eval_engine.py`

---

## 4. 1단계 — 학습과 ONNX 내보내기

```bash
python3 train_export.py            # 기본 3 epoch
```

모델 구조 (파라미터 206,922개 → FP32 가중치 약 808 KiB):

```text
Input 1×28×28
Conv(1→16, 3×3) → ReLU → MaxPool(2)       16×14×14
Conv(16→32, 3×3) → ReLU → MaxPool(2)      32×7×7
Flatten → Linear(1568→128) → ReLU → Linear(128→10)
```

- 파라미터의 97%가 첫 Linear(1568×128)에 있다 → engine 크기는 사실상 이 층이 결정한다.
- MNIST는 Google 미러에서 받는다. 학교망에서 막히면 `data/`에 4개 `.gz` 파일을 직접 넣는다.
- ONNX는 batch 차원만 가변(`batch`)으로 내보낸다. 최신 PyTorch의 새 exporter(dynamo)가 기본이어도 같은 결과가 나오도록 고전 exporter를 쓴다.

출력: `mnist_fp32.pth`, `mnist.onnx`, `calib_data.npy`(학습 이미지 1,024장), `test_data.npy`, `test_labels.npy`

예시 결과 (교수자 보드):

```text
Epoch 1/3  loss=0.3449
Epoch 2/3  loss=0.0787
Epoch 3/3  loss=0.0538
PyTorch FP32 accuracy: 98.38%
```

---

## 5. 2단계 — TensorRT engine 만들기 (세 precision, 같은 스크립트)

```bash
python3 build_engine.py fp32
python3 build_engine.py fp16
python3 build_engine.py int8
```

세 경우의 차이는 **builder flag 한 줄**뿐이다.

```python
if precision == "fp32":
    config.clear_flag(trt.BuilderFlag.TF32)     # 순수 FP32 기준
elif precision == "fp16":
    config.set_flag(trt.BuilderFlag.FP16)
elif precision == "int8":
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)       # INT8이 안 되는 layer는 FP16으로
    config.int8_calibrator = Calibrator()       # activation 범위 측정용
    config.set_calibration_profile(profile)
```

공통: 입력 shape 범위 `min 1×1×28×28 · opt 32×1×28×28 · max 128×1×28×28`, workspace 1 GiB.

### 5.1 FP32 — "기본값"이 사실은 TF32일 수 있다

Orin은 Ampere GPU라서 TensorRT가 FP32 layer에 **TF32**(가수 10 bit) Tensor Core를 기본으로 허용한다. 순수 FP32 기준을 만들려면 TF32를 꺼야 한다 (`trtexec`에서는 `--noTF32`).

| 형식 | 부호 | 지수 | 가수 | 비고 |
|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | 기준 |
| TF32 | 1 | 8 | 10 | Ampere Tensor Core 내부 연산 형식 |
| FP16 | 1 | 5 | 10 | 범위가 좁음 (최대 약 65,504) |
| INT8 | — | — | — | 정수 + scale |

### 5.2 FP16

`FP16` flag는 "FP16을 **써도 된다**"는 허용이다. TensorRT는 layer마다 실제로 더 빠른 kernel을 고르므로, 모든 layer가 FP16으로 실행된다는 뜻은 아니다.

### 5.3 INT8 — 왜 INT8만 calibrator가 필요한가

INT8은 실수를 정수로 근사한다 (TensorRT는 대칭 양자화):

$$
x_q = \operatorname{clip}\left(\operatorname{round}\left(\frac{x}{s}\right), -127, 127\right), \qquad \hat{x} = s\,x_q
$$

scale $s$를 정하려면 값의 범위를 알아야 한다.

- **weight**의 범위는 학습된 파일에 있으므로 바로 계산된다.
- **activation**(층 사이 중간 출력)의 범위는 **실제 입력을 넣어 봐야** 알 수 있다.

그래서 대표 입력(학습 이미지 1,024장)을 한 번 흘려 보내 층마다 범위를 재는 단계, 즉 **calibration**이 필요하다. 이것이 Post-Training Quantization(PTQ)이며, 모델을 다시 학습하는 QAT와 다르다. `IInt8EntropyCalibrator2`는 FP32 분포와 INT8 분포의 차이(KL divergence)가 가장 작아지는 범위를 고른다.

`build_engine.py`의 `Calibrator`가 하는 일은 단순하다 — **calib_data.npy를 32장씩 GPU에 복사하고 그 주소를 TensorRT에 넘긴다.** 범위 계산은 TensorRT가 한다.

```python
def get_batch(self, names):
    if self.i + self.batch > len(self.data):
        return None                                  # 데이터 끝
    self.buf.copy_(self.data[self.i:self.i + self.batch])
    self.i += self.batch
    return [self.buf.data_ptr()]                     # GPU 주소
```

- 결과는 `mnist_calib.cache`에 저장된다. **calibration 데이터를 바꿨다면 cache를 지우고** 다시 만든다.
- 참고: 이 방식(implicit quantization + calibrator)은 TensorRT 10.3에서 동작한다. 최신 TensorRT 문서는 ONNX에 Q/DQ 노드를 넣는 explicit quantization(예: NVIDIA Model Optimizer)을 권장하는 방향이므로, 버전을 올릴 때는 문서를 확인한다.

> **더 간단히?** `trtexec --onnx=mnist.onnx --int8`처럼 calibration 없이도 INT8 engine은 만들어진다. 하지만 이때 activation scale은 실제 데이터와 무관한 값이라 **속도 측정용일 뿐 정확도는 의미가 없다.** 정확도를 비교하려면 calibration이 꼭 필요하다.

### 5.4 Engine 크기 — 왜 정확히 4배가 아닌가

| Precision | Engine (예시) | 가중치만 계산 | 차이 (≈ 실행 정보 · metadata) |
|---|---:|---:|---:|
| FP32 | 901 KB | 808 KiB | ≈ 93 KB |
| FP16 | 514 KB | 404 KiB | ≈ 110 KB |
| INT8 | 318 KB | 202 KiB | ≈ 116 KB |

가중치 부분은 정확히 1 : 1/2 : 1/4로 줄지만, kernel 선택 · layer 정보 · scale 등 **약 100 KB의 고정 부분**이 있어 FP32 ÷ INT8 = 2.83배에 그친다. 큰 모델일수록 이 고정 부분의 비중이 작아져 4배에 가까워진다.

---

## 6. 3단계 — 정확도와 시간을 한 번에

```bash
python3 eval_engine.py mnist_fp32.engine mnist_fp16.engine mnist_int8.engine \
        --torch mnist_fp32.pth --out precision.json
```

- **정확도**: `test_data.npy` 10,000장 전체를 각 engine에 넣어 계산 (원본 실험에서 비어 있던 TensorRT engine 정확도).
- **시간**: 보드 기준 실험과 같은 방법 — 입력을 GPU에 미리 두고, B=1 · B=32 각각 warm-up 100회 + 3 trial × 200회, CUDA event로 median · p95.
- **`--torch`**: 같은 가중치를 PyTorch FP32로 실행한 값도 함께 → **런타임(PyTorch → TensorRT) 효과**와 **precision 효과**를 나눠 볼 수 있다.

출력 예 (형식):

```text
model            size KB   acc %    B=1 ms   B=1 p95   B=32 ms  B=32 img/s
PyTorch FP32         810   98.38    …
TensorRT FP32        901   …
TensorRT FP16        514   …
TensorRT INT8        318   …
```

---

## 7. (확인용) trtexec로 교차 측정

`trtexec`는 TensorRT에 포함된 벤치마크 도구다. 같은 engine으로 다시 재서 `eval_engine.py`의 값과 비교한다.

```bash
for p in fp32 fp16 int8; do
  /usr/src/tensorrt/bin/trtexec --loadEngine=mnist_$p.engine --shapes=input:32x1x28x28 \
      --warmUp=2000 --duration=10 2>&1 | grep -E "Throughput|Latency|GPU Compute Time"
done
```

`trtexec` 출력 읽는 법:

| 항목 | 의미 |
|---|---|
| **GPU Compute Time** | GPU에서 engine이 실행된 시간 → `eval_engine.py`의 median과 비교할 값 |
| **Latency** | Host→Device 복사 + GPU Compute + Device→Host 복사 |
| **Throughput (qps)** | 초당 **추론 호출** 수. 이미지/s = qps × batch |

- 평균(mean)보다 **median과 p99**를 본다 (`trtexec`가 함께 출력).
- `trtexec`로 engine을 직접 만들 수도 있다: FP32는 `--noTF32`, FP16은 `--fp16`. INT8은 위 5.3의 이유로 Python 스크립트를 쓴다.

---

## 8. 예시 결과 (교수자 보드, trtexec GPU Compute Time)

| Precision | B=1 GPU time | B=1 qps | B=32 GPU time | B=32 qps | B=32 images/s |
|---|---:|---:|---:|---:|---:|
| FP32 (TF32 허용 여부 미기록) | — | — | 0.254 ms | 3,858 | ≈ 123,500 |
| FP16 | 0.115 ms | 7,782 | 0.173 ms | 5,635 | ≈ 180,300 |
| INT8 (calibration) | 0.104 ms | 8,903 | 0.177 ms | 5,520 | ≈ 176,600 |

FP32 대비 (B=32, GPU time): FP16 ≈ 1.47배, INT8 ≈ 1.44배 빠름.

> 이 표는 원래 실험의 값이다. FP32 engine은 TF32가 허용된 기본 설정으로 만들어졌을 수 있고, B=1 FP32와 TensorRT engine 정확도는 측정되지 않았다. 위 6단계(`eval_engine.py`)로 빈칸을 채운다.

---

## 9. 해석

### 9.1 INT8이 항상 FP16보다 빠르지는 않다

B=32에서 FP16(0.173 ms)과 INT8(0.177 ms)은 거의 같다. 추론 시간은 대략

$$
T \approx T_{\text{compute}} + T_{\text{memory}} + T_{\text{kernel launch}} + T_{\text{quant/dequant}} + T_{\text{overhead}}
$$

이고, 이 모델은 **이미지 한 장에 약 1.2 M MAC (≈ 2.4 MFLOP)**으로 매우 작다 (보드 기준 실험의 CNN보다 약 150배 작음). 시간의 대부분이 연산이 아니라 kernel 실행 · 형식 변환 같은 고정 비용이라, bit 수를 줄여도 시간이 같은 비율로 줄지 않는다. FP16 Tensor Core kernel이 이미 충분히 빠르면 INT8의 이득은 더 작아진다.

$$
\text{Lower precision} \neq \text{Always lower latency}
$$

### 9.2 B=1의 10% 차이는 차이인가

B=1에서 INT8이 FP16보다 약 10% 빨랐지만, 둘 다 약 0.1 ms이다. 이 크기에서는 반복 측정의 편차와 비슷할 수 있으므로 **trial 3개의 값(`trial_median_ms`)이 겹치는지** 먼저 확인한다. 겹치면 "차이 없음"으로 보고한다.

### 9.3 Batch를 키우면

B=1 → B=32에서 한 번의 호출 시간은 약 1.5배만 늘지만 처리하는 이미지는 32배다. 작은 모델은 batch를 키워야 GPU가 일을 한다. 반대로 **카메라 한 장씩 처리하는 서비스는 B=1 지연**이 중요하다 — 어떤 batch를 볼지는 응용이 정한다.

### 9.4 Pareto 관점

정확도 · 지연 · 처리량 · 크기를 함께 놓고, 어느 지표에서도 다른 선택에 지지 않는 precision만 남긴다. 이 모델에서는 FP16과 INT8이 FP32를 거의 모든 지표에서 앞서고, FP16 vs INT8은 정확도 차이와 B=1 지연으로 결정된다.

---

## 10. 제출물

| 항목 | 내용 |
|---|---|
| `precision.json` | eval_engine.py 출력 그대로 |
| 비교표 | 정확도 · engine 크기 · B=1 median/p95 · B=32 images/s (PyTorch FP32 + TensorRT 3종) |
| trtexec 교차 확인 | 한 precision 이상, GPU Compute Time과 eval_engine.py median 비교 |
| 해석 | ① 런타임 효과(PyTorch → TRT FP32) ② precision 효과(FP32 → FP16 → INT8) ③ INT8 정확도 손실 |
| 환경 | 보드 기준 실험의 teamXX.json, TensorRT · torch 버전, 전력 모드 |

## 11. 문제 해결

| 증상 | 확인할 것 |
|---|---|
| `import tensorrt` 실패 | 가상환경을 `--system-site-packages` 없이 만들었음 → 다시 생성 |
| `torch.cuda.is_available()`가 False | pip로 torch를 덮어씀 → JetPack용 torch 재설치 |
| NumPy 관련 import 오류 | `pip install "numpy==1.26.4"` |
| INT8 정확도가 크게 떨어짐 | calibration 데이터가 학습 데이터와 다른 전처리(0~1 정규화)인지, 오래된 `mnist_calib.cache` 재사용 여부 |
| engine 로드 실패 | 다른 TensorRT 버전 · 다른 보드에서 만든 engine → 그 보드에서 다시 빌드 (engine은 이식되지 않음) |

## 참고

- NVIDIA, TensorRT Developer Guide — Working with Quantized Types / Reduced Precision (TF32, FP16, INT8, calibration)
- NVIDIA, TensorRT `trtexec` documentation
- B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018
- H. Wu et al., "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation," arXiv:2004.09602, 2020
