# Jetson Orin Nano에서 FP32 / FP16 / INT8 Quantization 실험

## 1. 실험 목적

Jetson Orin Nano에서 간단한 CNN을 직접 학습한 뒤, **같은 모델**을 TensorRT로 FP32 · FP16 · INT8 engine으로 변환하고 다음을 비교한다.

- 런타임 효과 (PyTorch → TensorRT)와 precision 효과 (FP32 → FP16 → INT8)
- precision flag로 **허용한** 형식과 TensorRT가 **실제로 사용한** 형식

- 분류 정확도 (test set 10,000장)
- TensorRT engine 크기
- 추론 시간 (median · p95) — **보드 기준 실험과 같은 CUDA event 방식**
- 처리량 (images/s)

```text
MNIST CNN 학습 (PyTorch, FP32)
  ↓  train_export.py
mnist.onnx  (+ calib_data.npy, test_data.npy)
  ↓  build_engine.py fp32 | fp16 | int8 | int8only   ← 모두 같은 스크립트
mnist_fp32.engine · mnist_fp16.engine · mnist_int8.engine · mnist_int8only.engine
  ↓  eval_engine.py
정확도 + 시간 + 크기 한 표 (precision.json)
  ↓  (확인용) trtexec
```

> 이 실험은 **보드 기준 실험(jetson_classification_benchmark.py)** 다음에 한다. 보드가 공정한지 먼저 확인해야 팀끼리 precision 결과를 비교할 수 있다.

---

## 2. 실험 환경

| 항목 | 값 |
|---|---|
| Device | Jetson Orin Nano (Super), GPU compute capability 8.7 (Ampere), 8 SM |
| 전력 모드 | MAXN SUPER + `jetson_clocks` (보드 기준 실험과 동일) |
| TensorRT | 10.3.0 (JetPack 6.x에 포함, 시스템에 설치되어 있음) |
| PyTorch | 2.8.0 Jetson용 CUDA 휠 (보드 기준 실험과 같은 버전) |
| NumPy | 2 미만 (1.26.x) |
| Python | 3.10 기반 가상환경 `~/quantization_lab/quant_env` |
| 작업 폴더 | `~/quantization_lab` |

이 실험에는 **TensorRT와 torch가 둘 다** 필요한데, 두 패키지는 설치된 위치가 다르다.

| 패키지 | 설치 위치 | 그래서 |
|---|---|---|
| TensorRT | 시스템 Python (`/usr/lib/python3.10/dist-packages`) | 가상환경을 `--system-site-packages`로 만들어야 보인다 |
| torch 2.8.0 | 가상환경 `quant_env` 안 | 가상환경을 켜지 않으면 보이지 않는다 |

따라서 **가상환경을 켜지 않고 실행하면 반드시 실패한다.** 이 실습에서 가장 흔한 실수다.

```text
ModuleNotFoundError: No module named 'torch'     ← 가상환경을 안 켜고 시스템 python3로 실행함
ModuleNotFoundError: No module named 'tensorrt'  ← 보드 기준 실험용 ~/benchmark/venv를 켜고 실행함
```

> [!IMPORTANT]
> **보드 기준 실험의 `~/benchmark/venv`를 재사용하지 않는다.** 그 가상환경은 시스템 패키지를 보지 못하도록 격리되어 있어 TensorRT가 없다. 이 실험용 가상환경을 3절대로 새로 만든다.

---

## 3. 가상환경 만들기 (조별 1회)

> [!IMPORTANT]
> 이 절의 명령은 모두 **sudo 없이** 실행한다. `sudo pip`, `sudo python`을 쓰면 시스템 Python에 설치되거나 파일이 root 소유가 되어 이후 설치가 깨진다.

### 3.1 다른 환경 끄기

프롬프트 맨 앞에 `(base)`, `(venv)` 같은 표시가 있으면 먼저 끈다.

```bash
conda deactivate 2>/dev/null; deactivate 2>/dev/null
```

프롬프트가 `hansungai@hansungai-desktop:~$`처럼 괄호 없이 시작하면 된다.

### 3.2 시스템에 TensorRT가 있는지 확인

```bash
/usr/bin/python3 -c "import tensorrt as trt; print(trt.__version__)"
```

`10.3.0`이 나오면 다음으로 넘어간다. 오류가 나면 **교수자에게 문의**한다 (보드에 TensorRT 설치가 필요하다).

### 3.3 작업 폴더와 가상환경 생성

```bash
mkdir -p ~/quantization_lab
cd ~/quantization_lab
/usr/bin/python3.10 -m venv --system-site-packages quant_env
```

- `--system-site-packages`를 빠뜨리면 TensorRT가 보이지 않는다. 빠뜨렸다면 `rm -rf quant_env` 후 다시 만든다.
- `ensurepip is not available` 오류가 나면 교수자에게 문의한다.

### 3.4 가상환경 켜기 ★

```bash
source ~/quantization_lab/quant_env/bin/activate
```

프롬프트 맨 앞에 **`(quant_env)`**가 붙어야 한다.

```text
(quant_env) hansungai@hansungai-desktop:~/quantization_lab$
```

> [!WARNING]
> 가상환경은 **터미널마다** 따로 켜야 한다. 새 터미널을 열거나, SSH로 다시 접속하거나, 보드를 재부팅하면 꺼진다. 그때마다 3.4의 명령을 다시 실행한다.
> **이후 모든 명령은 치기 전에 프롬프트에 `(quant_env)`가 있는지 먼저 본다.**

### 3.5 패키지 설치 (`(quant_env)` 상태에서)

```bash
# (1) pip 업데이트
python -m pip install --upgrade pip

# (2) Jetson 전용 CUDA torch — 보드 기준 실험과 같은 버전 (반드시 --no-deps 와 --index-url 을 함께)
python -m pip install --no-deps torch==2.8.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# (3) torch 의존성 + onnx (numpy 는 반드시 2 미만)
python -m pip install filelock "typing-extensions>=4.10" "sympy>=1.13.3" \
    networkx jinja2 fsspec "numpy<2" onnx
```

> [!CAUTION]
> - `pip install torch`를 index 지정 없이 실행하지 않는다. GPU를 쓰지 못하는 PyPI용 torch가 설치된다.
> - 이 가상환경에 `torchvision`, `ultralytics` 등 다른 패키지를 추가로 설치하지 않는다. torch를 PyPI판으로 바꿔 버릴 수 있다.
> - **pycuda는 설치하지 않는다.** GPU 버퍼는 PyTorch 텐서로 대신한다.

### 3.6 설치 확인

```bash
python -c "import sys, torch, tensorrt, numpy, onnx; print(sys.executable); print('torch', torch.__version__, torch.cuda.is_available(), '| tensorrt', tensorrt.__version__, '| numpy', numpy.__version__)"
```

다음처럼 나와야 한다.

```text
/home/hansungai/quantization_lab/quant_env/bin/python      ← quant_env 안의 python
torch 2.8.0 True | tensorrt 10.3.0 | numpy 1.26.4
```

- 첫 줄이 `/usr/bin/python3…`이면 가상환경이 꺼진 상태다 → 3.4로 돌아간다.
- `False`가 나오면 11절의 문제 해결을 본다.

### 3.7 실습 파일 놓기

실습 파일 3개를 `~/quantization_lab`에 둔다: `train_export.py`, `build_engine.py`, `eval_engine.py`

```powershell
# Windows PowerShell에서 (USB 연결 시)
scp train_export.py build_engine.py eval_engine.py hansungai@192.168.55.1:~/quantization_lab/
```

```bash
# 보드 기준 실험 폴더에 받아 두었다면 옮긴다
mv ~/benchmark/train_export.py ~/benchmark/build_engine.py ~/benchmark/eval_engine.py ~/quantization_lab/
```

- 2026-11-02 이후 배포본을 쓴다. 예전 파일이면 `build_engine.py`에 `int8only` 모드가 없고, `eval_engine.py`는 시간을 잘못 잰다 (11절).
- 세 파일은 **반드시 같은 폴더**에 둔다 (`eval_engine.py`가 `train_export.py`에서 모델 구조를 가져온다).
- 스크립트는 `mnist.onnx`, `calib_data.npy` 등을 **현재 폴더**에서 읽고 쓴다. 이후 모든 단계는 `~/quantization_lab`에서 실행한다.

### 3.8 다음에 다시 실습할 때

설치는 한 번만 하면 된다. 다음 시간에 터미널을 새로 열었다면 이 두 줄만 실행한다.

```bash
cd ~/quantization_lab
source quant_env/bin/activate        # 프롬프트에 (quant_env) 확인
```

---

## 4. 1단계 — 학습과 ONNX 내보내기

> [!IMPORTANT]
> **4~7절의 명령을 실행하기 전에 매번 확인한다.**
> 1. 프롬프트에 `(quant_env)`가 있다 → 없으면 `source ~/quantization_lab/quant_env/bin/activate`
> 2. 현재 폴더가 `~/quantization_lab`이다 → 아니면 `cd ~/quantization_lab`

```bash
python train_export.py            # 기본 3 epoch
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

첫 줄에 `Device: cuda Orin`이 나와야 GPU로 학습한 것이다. `Device: cpu`가 나오면 3.6으로 돌아가 확인한다.

예시 결과 (교수자 보드):

```text
Device: cuda Orin
Epoch 1/3  loss=0.3449
Epoch 2/3  loss=0.0787
Epoch 3/3  loss=0.0538
PyTorch FP32 accuracy: 98.38%
```

---

## 5. 2단계 — TensorRT engine 만들기 (네 가지 모드, 같은 스크립트)

```bash
python build_engine.py fp32
python build_engine.py fp16
python build_engine.py int8
python build_engine.py int8only
```

네 경우의 차이는 **builder flag**뿐이다.

| 모드 | 허용하는 precision | 결과 파일 |
|---|---|---|
| `fp32` | FP32만 (TF32도 끔) | `mnist_fp32.engine` |
| `fp16` | FP16 + FP32 | `mnist_fp16.engine` |
| `int8` | INT8 + FP16 + FP32 | `mnist_int8.engine` |
| `int8only` | INT8 + FP32 (TF32 끔) | `mnist_int8only.engine` |

```python
if precision == "fp32":
    config.clear_flag(trt.BuilderFlag.TF32)     # 순수 FP32 기준
elif precision == "fp16":
    config.set_flag(trt.BuilderFlag.FP16)
elif precision == "int8":
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)       # FP16도 허용 → 층마다 INT8과 FP16 중 빠른 쪽 선택
    config.int8_calibrator = Calibrator()       # activation 범위 측정용
    config.set_calibration_profile(profile)
elif precision == "int8only":
    config.set_flag(trt.BuilderFlag.INT8)       # INT8만 허용
    config.clear_flag(trt.BuilderFlag.TF32)     # INT8이 안 되는 층은 순수 FP32
    config.int8_calibrator = Calibrator()
    config.set_calibration_profile(profile)
```

공통: 입력 shape 범위 `min 1×1×28×28 · opt 32×1×28×28 · max 128×1×28×28`, workspace 1 GiB.

빌드가 끝나면 스크립트가 **층마다 실제로 선택된 가중치 형식**을 출력한다 (`Float`=FP32, `Half`=FP16, `Int8`=INT8). 이 출력을 반드시 기록한다. flag는 "써도 된다"는 **허용**일 뿐이라, 실제로 어떤 precision이 쓰였는지는 이 출력으로만 알 수 있다 (5.3).

빌드 중 나오는 다음 메시지는 정상이다.

| 메시지 | 의미 |
|---|---|
| `DLA requests all profiles have same min, max, and opt value. All dla layers are falling back to GPU` | DLA(별도 가속기)용 안내. Orin Nano에는 DLA가 없고 이 실험은 GPU만 쓴다 → 무시 |
| `DeprecationWarning: ... Superseded by explicit quantization` | calibrator 방식이 최신 버전에서 권장되지 않는다는 안내. TensorRT 10.3에서는 정상 동작 (5.3 참고) |

### 5.1 FP32 — "기본값"이 사실은 TF32일 수 있다

Orin은 Ampere GPU라서 TensorRT가 FP32 layer에 **TF32**(가수 10 bit) Tensor Core를 기본으로 허용한다. 순수 FP32 기준을 만들려면 TF32를 꺼야 한다 (`trtexec`에서는 `--noTF32`).

| 형식 | 부호 | 지수 | 가수 | 비고 |
|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | 기준 |
| TF32 | 1 | 8 | 10 | Ampere Tensor Core 내부 연산 형식 |
| FP16 | 1 | 5 | 10 | 범위가 좁음 (최대 약 65,504) |
| INT8 | — | — | — | 정수 + scale |

### 5.2 FP16

`FP16` flag는 "FP16을 **써도 된다**"는 허용이다. TensorRT는 빌드할 때 허용된 precision의 kernel을 층마다 실제로 실행해 보고(tactic 선택) 가장 빠른 것을 고른다. 따라서 모든 layer가 FP16으로 실행된다는 뜻은 아니다.

### 5.3 INT8 — 허용과 사용은 다르다

INT8은 실수를 정수로 근사한다 (TensorRT는 대칭 양자화):

$$
x_q = \mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right), -127, 127\right), \qquad \hat{x} = s\,x_q
$$

scale $s$를 정하려면 값의 범위를 알아야 한다.

- **weight**의 범위는 학습된 파일에 있으므로 바로 계산된다.
- **activation**(층 사이 중간 출력)의 범위는 **실제 입력을 넣어 봐야** 알 수 있다.

그래서 대표 입력(학습 이미지 1,024장)을 한 번 흘려 보내 층마다 범위를 재는 단계, 즉 **calibration**이 필요하다. 이것이 Post-Training Quantization(PTQ)이며, 모델을 다시 학습하는 QAT와 다르다. `IInt8EntropyCalibrator2`는 FP32 분포와 INT8 분포의 차이(KL divergence)가 가장 작아지는 범위를 고른다.

`build_engine.py`의 `Calibrator`가 하는 일은 단순하다 — **calib_data.npy를 32장씩 GPU에 복사하고 그 주소를 TensorRT에 넘긴다.** 범위 계산은 TensorRT가 한다. (GPU 버퍼를 torch 텐서로 만들기 때문에 이 스크립트에도 torch가 필요하다.)

```python
def get_batch(self, names):
    if self.i + self.batch > len(self.data):
        return None                                  # 데이터 끝
    self.buf.copy_(self.data[self.i:self.i + self.batch])
    self.i += self.batch
    return [self.buf.data_ptr()]                     # GPU 주소
```

**`int8`과 `int8only`를 둘 다 만드는 이유.** INT8을 쓰려면 층 경계에서 형식 변환(quantize/dequantize)이 필요하다. 이 모델처럼 층이 작으면 INT8 kernel이 FP16 Tensor Core kernel보다 빠르지 않을 수 있다. 그러면 `int8` 모드에서는 TensorRT가 **INT8을 허용받고도 FP16을 고른다.** 교수자 보드에서 실제로 그랬다.

| 모드 | Conv1 · Conv2 · FC1 가중치 (교수자 보드) | 뜻 |
|---|---|---|
| `int8` | `Half` | calibration은 했지만 INT8은 쓰이지 않음 → 사실상 FP16 engine |
| `int8only` | `Int8` (마지막 FC2는 `Float`) | 실제 INT8 engine |

빌드 출력에서 자기 보드의 결과를 확인하고, 결과 해석에서 "`int8` engine이 실제로 INT8이었는가"를 반드시 밝힌다. 이미 만든 engine은 다음으로도 확인할 수 있다.

```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=mnist_int8.engine --profilingVerbosity=detailed \
    --dumpLayerInfo --skipInference 2>&1 | grep -oE '"Type": "[A-Za-z0-9]+", "Count": [0-9]+'
```

- 결과는 `mnist_calib.cache`에 저장되고, `int8`과 `int8only`가 같은 cache를 함께 쓴다. **calibration 데이터를 바꿨다면 cache를 지우고** 다시 만든다.
- 참고: 이 방식(implicit quantization + calibrator)은 TensorRT 10.3에서 동작한다. 최신 TensorRT 문서는 ONNX에 Q/DQ 노드를 넣는 explicit quantization(예: NVIDIA Model Optimizer)을 권장하는 방향이므로, 버전을 올릴 때는 문서를 확인한다.

> **더 간단히?** `trtexec --onnx=mnist.onnx --int8`처럼 calibration 없이도 INT8 engine은 만들어진다. 하지만 이때 activation scale은 실제 데이터와 무관한 값이라 **속도 측정용일 뿐 정확도는 의미가 없다.** 정확도를 비교하려면 calibration이 꼭 필요하다. 또 `trtexec --shapes=...`로 만든 engine은 batch가 하나로 고정되어 6절의 B=1 · B=32 측정에 쓸 수 없다. engine은 `build_engine.py`로 만든다.

### 5.4 Engine 크기 — 왜 정확히 4배가 아닌가

| 모드 | Engine (교수자 보드) | 가중치만 계산 | 차이 (≈ 실행 정보 · metadata) |
|---|---:|---:|---:|
| `fp32` | 916 KB | 808 KiB | ≈ 108 KB |
| `fp16` | 516 KB | 404 KiB | ≈ 112 KB |
| `int8` | 506 KB | 404 KiB (실제로는 FP16) | ≈ 102 KB |
| `int8only` | 316 KB | ≈ 206 KiB (INT8 + FC2 FP32) | ≈ 110 KB |

가중치 부분은 1 : 1/2 : 1/4로 줄지만, kernel 선택 · layer 정보 · scale 등 **약 100 KB의 고정 부분**이 있어 FP32 ÷ INT8 = 2.9배에 그친다. 큰 모델일수록 이 고정 부분의 비중이 작아져 4배에 가까워진다. `int8`이 `fp16`과 크기가 같다는 것도 "INT8이 쓰이지 않았다"는 증거다.

---

## 6. 3단계 — 정확도와 시간을 한 번에

```bash
python eval_engine.py mnist_fp32.engine mnist_fp16.engine mnist_int8.engine mnist_int8only.engine \
        --torch mnist_fp32.pth --out precision.json
```

- **정확도**: `test_data.npy` 10,000장 전체를 각 engine에 넣어 계산.
- **시간**: 보드 기준 실험과 같은 방법 — 입력을 GPU에 미리 두고, B=1 · B=32 각각 warm-up 100회 + 3 trial × 200회, CUDA event로 median · p95.
- **`--torch`**: 같은 가중치를 PyTorch FP32로 실행한 값도 함께 → **런타임(PyTorch → TensorRT) 효과**와 **precision 효과**를 나눠 볼 수 있다. 이 옵션은 같은 폴더의 `train_export.py`에서 모델 구조를 가져온다.

> [!NOTE]
> **측정 방법에서 주의한 점.** 이 모델은 GPU 연산이 0.1 ms 수준이라, 측정 구간 안에 Python 작업이 조금만 끼어도 GPU가 그동안 놀고 CUDA event가 **연산이 아니라 호스트(Python) 시간**을 잰다. 그래서 `eval_engine.py`는
> 1. TensorRT를 **전용(non-default) CUDA stream**에서 실행하고 (기본 stream이면 TensorRT가 매 호출마다 추가 동기화를 넣는다),
> 2. shape 설정 · 출력 버퍼 할당 · 주소 지정은 **측정 구간 밖에서 한 번만** 하고, 측정 구간에서는 `execute_async_v3()`만 호출한다.
>
> 실행 중 `Using default stream in enqueueV3()` 경고가 나오면 **예전 `eval_engine.py`**다. 이 경우 모든 engine의 시간이 B·precision과 상관없이 약 0.2 ms로 비슷하게 나와 비교가 불가능하다 → 최신 파일로 교체한다.

출력 예 (교수자 보드):

```text
model             size KB   acc %    B=1 ms   B=1 p95   B=32 ms  B=32 img/s
PyTorch FP32          812   98.29    0.8490    0.8736    0.8817       36292
TensorRT FP32         916   98.29    0.1505    0.1630    0.2172      147308
TensorRT FP16         516   98.29    0.1102    0.1236    0.1317      242925
TensorRT INT8         506   98.29    0.1076    0.1185    0.1269      252080
TensorRT INT8ONLY     316   94.26    0.1015    0.1118    0.1323      241955
```

---

## 7. (확인용) trtexec로 교차 측정

`trtexec`는 TensorRT에 포함된 벤치마크 도구다 (Python이 아니라 실행 파일이므로 가상환경과 무관하게 동작한다). 같은 engine으로 다시 재서 `eval_engine.py`의 값과 비교한다.

```bash
cd ~/quantization_lab
for p in fp32 fp16 int8 int8only; do
  echo "== $p"
  /usr/src/tensorrt/bin/trtexec --loadEngine=mnist_$p.engine --shapes=input:32x1x28x28 \
      --warmUp=2000 --duration=10 2>&1 | grep -E "Throughput|GPU Compute Time"
done
```

`trtexec` 출력 읽는 법:

| 항목 | 의미 |
|---|---|
| **GPU Compute Time** | GPU에서 engine이 실행된 시간 → `eval_engine.py`의 median과 비교할 값 |
| **Latency** | Host→Device 복사 + GPU Compute + Device→Host 복사 |
| **Throughput (qps)** | 초당 **추론 호출** 수. 이미지/s = qps × batch |

- 평균(mean)보다 **median과 p99**를 본다 (`trtexec`가 함께 출력).
- `eval_engine.py` 값이 `trtexec`보다 약간(교수자 보드에서 0~0.02 ms) 크게 나올 수 있다. `eval_engine.py`는 Python에서 한 번씩 실행을 요청하고, `trtexec`는 C++에서 쉬지 않고 이어서 요청하기 때문이다. 0.1 ms 수준의 작은 모델에서만 눈에 띄는 차이다.

---

## 8. 예시 결과 (교수자 보드, 2026-11-02)

환경: Jetson Orin Nano Super, TensorRT 10.3.0, torch 2.8.0, B=32 기준.

| 모드 | Engine | 정확도 | eval B=1 | eval B=32 | trtexec B=32 GPU Compute (median) | 실제 가중치 형식 |
|---|---:|---:|---:|---:|---:|---|
| PyTorch FP32 | 812 KB | 98.29% | 0.849 ms | 0.882 ms | — | FP32 |
| `fp32` | 916 KB | 98.29% | 0.151 ms | 0.217 ms | 0.217 ms | FP32 |
| `fp16` | 516 KB | 98.29% | 0.110 ms | 0.132 ms | 0.114 ms | FP16 |
| `int8` | 506 KB | 98.29% | 0.108 ms | 0.127 ms | 0.111 ms | **FP16** (INT8 미사용) |
| `int8only` | 316 KB | **94.26%** | 0.102 ms | 0.132 ms | 0.113 ms | INT8 (FC2만 FP32) |

> `int8only`의 trtexec 값은 batch 32 고정 shape로 따로 만든 engine에서 잰 것이다. 다른 값은 모두 `build_engine.py`로 만든 같은 engine에서 잰 것이다.

---

## 9. 해석

### 9.1 런타임 효과가 precision 효과보다 크다

| 비교 | B=1 | B=32 |
|---|---:|---:|
| PyTorch FP32 → TensorRT FP32 (**런타임**) | 0.849 → 0.151 ms, **5.6배** | 0.882 → 0.217 ms, **4.1배** |
| TensorRT FP32 → FP16 (**precision**) | 0.151 → 0.110 ms, 1.4배 | 0.217 → 0.132 ms, 1.6배 |

precision을 바꾸기도 전에, 런타임을 PyTorch에서 TensorRT로 바꾸는 것만으로 4~6배 빨라졌다. PyTorch(eager)는 층마다 GPU kernel을 Python에서 하나씩 실행시키므로, 이렇게 작은 모델에서는 그 호출 비용이 전체 시간을 지배한다(B=1과 B=32의 시간이 거의 같은 것이 그 증거). TensorRT는 층을 합치고(Conv+ReLU fusion) 호출을 한 번으로 묶어 이 비용을 없앤다.

### 9.2 INT8을 허용해도 쓰이지 않을 수 있다

`int8` engine은 calibration까지 했지만 모든 큰 층이 FP16으로 실행됐다. 크기(506 KB ≈ 516 KB) · 정확도(98.29%) · 속도가 모두 `fp16`과 같다. TensorRT는 허용된 precision 중 **이 보드에서 더 빠른 kernel**을 고르고, 이 모델에서는 그것이 FP16이었다.

$$
\text{precision flag} = \text{허용} \neq \text{사용}
$$

### 9.3 INT8이 항상 더 빠르지는 않다

INT8만 허용한 `int8only`도 B=32에서 FP16과 같은 시간이다 (trtexec 0.113 ms vs 0.114 ms). 추론 시간은 대략

$$
T \approx T_{\text{compute}} + T_{\text{memory}} + T_{\text{kernel launch}} + T_{\text{quant/dequant}} + T_{\text{overhead}}
$$

이고, 이 모델은 **이미지 한 장에 약 1.2 M MAC (≈ 2.4 MFLOP)**으로 매우 작다 (보드 기준 실험의 CNN은 약 184 M MAC로 약 150배). 시간의 대부분이 연산이 아니라 kernel 실행 · 형식 변환 같은 고정 비용이라, bit 수를 줄여도 시간이 같은 비율로 줄지 않는다.

$$
\text{Lower precision} \neq \text{Always lower latency}
$$

### 9.4 INT8의 대가 — 정확도

`int8only`는 정확도가 98.29% → 94.26%로 **4.0%p** 떨어졌다. FP16은 손실이 없었다. PTQ의 정확도 손실은 calibration 방법과 데이터에 따라 달라진다. 원인을 확인하려면 다음을 바꿔 본다 (심화).

- calibration 이미지 수 (`python train_export.py --calib 4096` → `rm mnist_calib.cache` → 다시 빌드)
- calibrator 종류 (`IInt8EntropyCalibrator2` ↔ `IInt8MinMaxCalibrator`)

### 9.5 작은 차이는 차이인가

B=1에서 `int8only`(0.102 ms)가 `fp16`(0.110 ms)보다 약 8% 빨랐지만, 둘 다 약 0.1 ms이다. 이 크기에서는 반복 측정의 편차와 비슷할 수 있으므로 **trial 3개의 값(`precision.json`의 `trial_median_ms`)이 겹치는지** 먼저 확인한다. 겹치면 "차이 없음"으로 보고한다.

### 9.6 Batch를 키우면

TensorRT FP16은 B=1 → B=32에서 한 번의 호출 시간이 0.110 → 0.132 ms로 약 1.2배만 늘지만, 처리하는 이미지는 32배다. 작은 모델은 batch를 키워야 GPU가 일을 한다. 반대로 **카메라 한 장씩 처리하는 서비스는 B=1 지연**이 중요하다 — 어떤 batch를 볼지는 응용이 정한다.

### 9.7 Pareto 관점

정확도 · 지연 · 처리량 · 크기를 함께 놓고, 어느 지표에서도 다른 선택에 지지 않는 모드만 남긴다.

- `fp16`은 `fp32`보다 정확도는 같고, 더 빠르고, 더 작다 → `fp32`는 탈락.
- `int8`은 사실상 `fp16`과 같은 engine이다.
- `int8only`는 **크기**에서만 이기고, 속도는 같고, 정확도는 낮다.

이 모델에서는 **FP16이 가장 균형 잡힌 선택**이다. 저장 공간이 가장 중요한 조건일 때만 `int8only`가 의미가 있다. 모델이 커져 연산이 시간을 지배하면 INT8의 속도 이득이 나타날 수 있으므로, 이 결론은 "이 모델, 이 보드"에 대한 것이다.

---

## 10. 제출물

| 항목 | 내용 |
|---|---|
| `precision.json` | eval_engine.py 출력 그대로 (`int8only` 포함 5개 모델) |
| 층별 precision | `build_engine.py int8`과 `int8only` 실행 시 출력된 "실제로 선택된 가중치 형식" |
| 비교표 | 정확도 · engine 크기 · B=1 median/p95 · B=32 images/s (PyTorch FP32 + TensorRT 4종) |
| trtexec 교차 확인 | 한 모드 이상, GPU Compute Time과 eval_engine.py median 비교 |
| 해석 | ① 런타임 효과(PyTorch → TRT FP32) ② precision 효과(FP32 → FP16 → INT8) ③ `int8` engine이 실제로 INT8이었는가 ④ INT8 정확도 손실 |
| 환경 | 보드 기준 실험의 teamXX.json, 3.6 설치 확인 출력(torch 2.8.0 · TensorRT 10.3.0), 전력 모드 |

## 11. 문제 해결

**대부분의 오류는 가상환경이 꺼진 상태에서 실행해서 생긴다.** 먼저 프롬프트에 `(quant_env)`가 있는지, 현재 폴더가 `~/quantization_lab`인지 확인한다.

| 증상 | 원인 | 해결 |
|---|---|---|
| `No module named 'torch'` | 가상환경을 켜지 않아 시스템 Python으로 실행됨 | `source ~/quantization_lab/quant_env/bin/activate` 후 다시 실행 |
| `No module named 'tensorrt'` | ① `~/benchmark/venv` 등 다른 가상환경이 켜져 있음 ② `quant_env`를 `--system-site-packages` 없이 만듦 | ① `deactivate` 후 3.4 ② `rm -rf quant_env` 후 3.3부터 다시 |
| `python: command not found` | 가상환경이 꺼져 있음 (시스템에는 `python` 명령이 없을 수 있음) | 3.4 |
| `torch.cuda.is_available()`가 False, 또는 학습 시 `Device: cpu` | PyPI용 torch가 설치됨 (index 없이 `pip install torch`, 또는 다른 패키지 설치가 torch를 바꿈) | 3.5의 (2)에 `--force-reinstall`을 붙여 다시 실행 |
| `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` | NumPy 2가 설치됨 | `python -m pip install "numpy<2"` |
| `FileNotFoundError: mnist.onnx` / `calib_data.npy` | 다른 폴더에서 실행했거나 1단계를 건너뜀 | `cd ~/quantization_lab` 후 4절부터 |
| `No module named 'train_export'` (6절) | 세 파일이 같은 폴더에 없음 | 3.7 |
| `argument precision: invalid choice: 'int8only'` | 예전 `build_engine.py` | 최신 파일로 교체 |
| 6절에서 `Using default stream in enqueueV3()` 경고, 모든 시간이 약 0.2 ms로 비슷함 | 예전 `eval_engine.py` | 최신 파일로 교체 (6절 NOTE) |
| 6절에서 `RuntimeError: The size of tensor a (32) must match the size of tensor b (128)` | `trtexec --shapes=...`로 만든 batch 고정 engine을 넣음 | `python build_engine.py <모드>`로 다시 만든다 |
| `DLA requests all profiles ...` 경고 | Orin Nano에는 DLA가 없음 | 무시 (5절 표) |
| 설치 중 `Permission denied` | 이전에 `sudo pip`를 써서 root 소유 파일이 생김 | 교수자에게 문의 |
| INT8 정확도가 크게 떨어짐 | calibration 데이터 전처리(0~1 정규화)가 다르거나, 오래된 `mnist_calib.cache` 재사용 | `rm mnist_calib.cache` 후 다시 빌드. 그래도 낮으면 9.4 |
| engine 로드 실패 | 다른 TensorRT 버전 · 다른 보드에서 만든 engine | 그 보드에서 다시 빌드 (engine은 이식되지 않음) |

## 참고

- NVIDIA, TensorRT Developer Guide — Working with Quantized Types / Reduced Precision (TF32, FP16, INT8, calibration)
- NVIDIA, TensorRT `trtexec` documentation
- B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018
- H. Wu et al., "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation," arXiv:2004.09602, 2020
