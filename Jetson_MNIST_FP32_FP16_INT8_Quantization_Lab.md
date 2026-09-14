# Jetson Orin Nano에서 FP32 / FP16 / INT8 Quantization 실험

## 1. 실험 목적

본 실험의 목적은 NVIDIA Jetson Orin Nano에서 간단한 CNN 모델을 직접 학습한 뒤, 동일한 모델을 TensorRT를 이용해 FP32, FP16, INT8 precision으로 변환하고 다음 항목을 비교하는 것이다.

- 분류 정확도
- TensorRT engine 크기
- inference latency
- GPU compute time
- throughput

전체 실험 흐름은 다음과 같다.

```text
Jetson
  ↓
Python virtual environment
  ↓
PyTorch + CUDA
  ↓
MNIST CNN 학습 (FP32)
  ↓
PyTorch model (.pth)
  ↓
ONNX export
  ↓
TensorRT
  ├─ FP32 engine
  ├─ FP16 engine
  └─ INT8 engine (PTQ calibration)
  ↓
Latency / Throughput / Engine size 비교
```

---

# 2. 실험 환경

본 실험에서 확인된 주요 환경은 다음과 같다.

```text
Device: NVIDIA Jetson Orin Nano
GPU Compute Capability: 8.7
GPU SMs: 8
GPU Memory: 약 7.6 GB
TensorRT: 10.3.0
Python: 3.10
CUDA: JetPack에 포함된 CUDA 환경
```

---

# 3. Python 가상환경 구성

Jetson에서는 Anaconda Python보다 JetPack에 포함된 시스템 Python을 기반으로 가상환경을 구성하는 것이 TensorRT, CUDA와의 호환성 측면에서 편리하다.

```bash
cd ~
mkdir -p quantization_lab
cd quantization_lab
```

가상환경 생성:

```bash
/usr/bin/python3 -m venv --system-site-packages quant_env
```

활성화:

```bash
source quant_env/bin/activate
```

정상적으로 활성화되면 프롬프트는 다음과 비슷하다.

```text
(quant_env) hansungai@ubuntu:~/quantization_lab$
```

TensorRT 확인:

```bash
python -c "import tensorrt as trt; print(trt.__version__)"
```

실험 환경에서는 다음과 같이 확인되었다.

```text
10.3.0
```

---

# 4. PyTorch 및 기본 패키지

Jetson에서는 일반적인 x86 환경과 달리 JetPack과 호환되는 NVIDIA용 PyTorch wheel을 사용하는 것이 안전하다.

설치 후 CUDA 확인:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch CUDA:", torch.version.cuda)
PY
```

핵심은 다음과 같이 CUDA가 활성화되는 것이다.

```text
CUDA available: True
GPU: Orin
```

NumPy는 PyTorch binary compatibility 문제를 피하기 위해 1.x 계열을 사용하였다.

```bash
pip install --force-reinstall "numpy==1.26.4"
```

ONNX 설치:

```bash
pip install onnx
```

---

# 5. MNIST CNN 학습

## 5.1 모델 구조

본 실험에서는 연산량이 작고 구조가 단순한 CNN을 사용하였다.

```text
Input: 1 × 28 × 28

Conv2d(1 → 16, 3×3)
ReLU
MaxPool2d(2)

Conv2d(16 → 32, 3×3)
ReLU
MaxPool2d(2)

Flatten
Linear(32×7×7 → 128)
ReLU
Linear(128 → 10)
```

이 모델은 양자화 전후의 성능 차이를 관찰하기에 충분히 단순하면서도 CNN의 Conv, ReLU, Pooling, Fully Connected 연산을 포함한다.

---

## 5.2 학습 코드

`torchvision` 의존성을 줄이기 위해 MNIST 파일을 직접 다운로드하고 NumPy/PyTorch로 읽었다.

```python
import os
import gzip
import struct
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. MNIST Download
# ============================================================

os.makedirs("data", exist_ok=True)

base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

files = {
    "train-images-idx3-ubyte.gz":
        base_url + "train-images-idx3-ubyte.gz",

    "train-labels-idx1-ubyte.gz":
        base_url + "train-labels-idx1-ubyte.gz",

    "t10k-images-idx3-ubyte.gz":
        base_url + "t10k-images-idx3-ubyte.gz",

    "t10k-labels-idx1-ubyte.gz":
        base_url + "t10k-labels-idx1-ubyte.gz"
}


for filename, url in files.items():

    path = os.path.join("data", filename)

    if not os.path.exists(path):

        print("Downloading:", filename)

        urllib.request.urlretrieve(
            url,
            path
        )


# ============================================================
# 3. MNIST Loader
# ============================================================

def load_images(path):

    with gzip.open(path, "rb") as f:

        magic, num, rows, cols = struct.unpack(
            ">IIII",
            f.read(16)
        )

        data = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        data = data.reshape(
            num,
            1,
            rows,
            cols
        )

        data = data.astype(
            np.float32
        ) / 255.0

    return data


def load_labels(path):

    with gzip.open(path, "rb") as f:

        magic, num = struct.unpack(
            ">II",
            f.read(8)
        )

        labels = np.frombuffer(
            f.read(),
            dtype=np.uint8
        ).astype(np.int64)

    return labels


train_x = load_images(
    "data/train-images-idx3-ubyte.gz"
)

train_y = load_labels(
    "data/train-labels-idx1-ubyte.gz"
)

test_x = load_images(
    "data/t10k-images-idx3-ubyte.gz"
)

test_y = load_labels(
    "data/t10k-labels-idx1-ubyte.gz"
)


print("Train:", train_x.shape)
print("Test :", test_x.shape)


train_dataset = TensorDataset(
    torch.from_numpy(train_x),
    torch.from_numpy(train_y)
)

test_dataset = TensorDataset(
    torch.from_numpy(test_x),
    torch.from_numpy(test_y)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)


# ============================================================
# 4. CNN
# ============================================================

class SmallCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                10
            )
        )


    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


model = SmallCNN().to(device)


# ============================================================
# 5. Training
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


epochs = 3


for epoch in range(epochs):

    model.train()

    running_loss = 0.0


    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(
            output,
            y
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss={running_loss / len(train_loader):.4f}"
    )


# ============================================================
# 6. Evaluation
# ============================================================

model.eval()

correct = 0
total = 0


with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)
        y = y.to(device)

        output = model(x)

        pred = output.argmax(
            dim=1
        )

        correct += (
            pred == y
        ).sum().item()

        total += y.size(0)


accuracy = correct / total


print(
    f"FP32 Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 7. Save Model and Data
# ============================================================

torch.save(
    model.state_dict(),
    "mnist_fp32.pth"
)

np.save(
    "calibration_data.npy",
    train_x[:1024]
)

np.save(
    "test_data.npy",
    test_x
)

np.save(
    "test_labels.npy",
    test_y
)
```

---

## 5.3 학습 결과

실제 Jetson에서 얻은 학습 결과:

```text
Device: cuda
GPU: Orin

Train: (60000, 1, 28, 28)
Test : (10000, 1, 28, 28)

Epoch 1/3, Loss=0.3449
Epoch 2/3, Loss=0.0787
Epoch 3/3, Loss=0.0538

FP32 Accuracy: 98.38%
```

따라서 기준 모델의 정확도는

\[
\boxed{98.38\%}
\]

이다.

---

# 6. ONNX로 변환

TensorRT는 PyTorch의 `.pth` 파일을 직접 사용하는 것이 아니라, 일반적으로 ONNX를 중간 표현으로 사용한다.

전체 변환 과정은

```text
PyTorch
  ↓
ONNX
  ↓
TensorRT Engine
```

이다.

## 6.1 ONNX export 코드

```python
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = SmallCNN()

state_dict = torch.load(
    "mnist_fp32.pth",
    map_location="cpu"
)

model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(
    1, 1, 28, 28
)

torch.onnx.export(
    model,
    dummy_input,
    "mnist.onnx",

    input_names=["input"],
    output_names=["output"],

    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    },

    opset_version=17
)

print("Saved: mnist.onnx")
```

ONNX 검증:

```bash
python - <<'PY'
import onnx

model = onnx.load("mnist.onnx")
onnx.checker.check_model(model)

print("ONNX model OK")
PY
```

---

# 7. FP32 / FP16 / INT8의 의미

## 7.1 FP32

FP32는 IEEE 754 single precision floating-point를 사용한다.

하나의 값이 32 bit를 사용하므로, 기준 precision으로 볼 수 있다.

```text
FP32
= 1 sign bit
+ 8 exponent bits
+ 23 fraction bits
```

본 실험에서는 FP32로 학습한 PyTorch 모델을 그대로 TensorRT engine으로 변환하였다.

```bash
/usr/src/tensorrt/bin/trtexec \
--onnx=mnist.onnx \
--minShapes=input:1x1x28x28 \
--optShapes=input:32x1x28x28 \
--maxShapes=input:128x1x28x28 \
--saveEngine=mnist_fp32.engine
```

즉 별도의 precision 옵션을 주지 않으면 FP32가 기본 기준이 된다.

---

## 7.2 FP16

FP16은 16-bit floating point 표현이다.

```text
FP16
= 1 sign bit
+ 5 exponent bits
+ 10 fraction bits
```

FP32 대비 메모리 사용량을 줄이고, Jetson GPU의 Tensor Core 등 저정밀 연산 하드웨어를 사용할 수 있다.

TensorRT에서는 `--fp16` 옵션만 추가하면 된다.

```bash
/usr/src/tensorrt/bin/trtexec \
--onnx=mnist.onnx \
--fp16 \
--minShapes=input:1x1x28x28 \
--optShapes=input:32x1x28x28 \
--maxShapes=input:128x1x28x28 \
--saveEngine=mnist_fp16.engine
```

핵심은 다음 한 줄이다.

```text
--fp16
```

이 옵션은 TensorRT builder에게 FP16 precision을 사용 가능한 연산에서 활용하도록 허용한다.

중요한 점은 모든 layer가 무조건 FP16으로 실행된다는 의미가 아니라는 것이다. TensorRT는 지원되는 kernel과 성능을 고려하여 실제 실행 precision을 선택할 수 있다.

---

# 8. INT8 Post-Training Quantization

## 8.1 INT8 양자화 개념

INT8 quantization은 실수 값을 8-bit 정수 값으로 근사한다.

대표적인 symmetric quantization은 다음과 같이 표현할 수 있다.

\[
x_q
=
\operatorname{clip}
\left(
\operatorname{round}
\left(
\frac{x}{s}
\right),
-128,
127
\right)
\]

여기서

- \(x\): 원래 FP32 값
- \(x_q\): INT8 정수 값
- \(s\): scale factor

이다.

복원 시에는

\[
\hat{x}=s x_q
\]

를 사용한다.

즉 INT8에서는 실수 값을 단순히 8-bit로 자르는 것이 아니라, 실수 범위를 정수 범위에 매핑할 scale을 결정해야 한다.

---

## 8.2 왜 Calibration이 필요한가?

Weight뿐 아니라 activation도 INT8로 변환하려면 각 activation의 실제 값 범위를 알아야 한다.

예를 들어 activation 값이 대략

\[
[-3.2, 2.8]
\]

범위에서 발생한다면, 이를 INT8 범위

\[
[-128,127]
\]

에 적절히 대응시키는 scale을 정해야 한다.

이를 위해 대표적인 입력 데이터를 모델에 통과시키는 과정을 calibration이라고 한다.

본 실험에서는 학습 데이터 중 1024장을 calibration에 사용하였다.

```python
np.save(
    "calibration_data.npy",
    train_x[:1024]
)
```

이 방식은 **Post-Training Quantization (PTQ)** 이다.

```text
FP32 training
     ↓
trained model
     ↓
representative calibration data
     ↓
INT8 range / scale estimation
     ↓
INT8 TensorRT engine
```

즉 모델을 INT8 상태로 다시 학습하는 QAT(Quantization-Aware Training)과는 다르다.

---

# 9. TensorRT INT8 Calibrator

INT8 engine 생성을 위해 TensorRT의 calibration API를 사용하였다.

```python
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class MNISTCalibrator(trt.IInt8EntropyCalibrator2):

    def __init__(
        self,
        calibration_file="calibration_data.npy",
        batch_size=32,
        cache_file="mnist_calibration.cache"
    ):
        super().__init__()

        self.batch_size = batch_size
        self.cache_file = cache_file

        self.data = np.load(
            calibration_file
        ).astype(np.float32)

        self.data = np.ascontiguousarray(
            self.data
        )

        self.current_index = 0

        sample_bytes = self.data[0].nbytes

        self.device_input = cuda.mem_alloc(
            self.batch_size * sample_bytes
        )


    def get_batch_size(self):
        return self.batch_size


    def get_batch(self, names):

        if (
            self.current_index
            + self.batch_size
            > len(self.data)
        ):
            return None

        batch = self.data[
            self.current_index:
            self.current_index
            + self.batch_size
        ]

        batch = np.ascontiguousarray(
            batch
        )

        cuda.memcpy_htod(
            self.device_input,
            batch
        )

        self.current_index += self.batch_size

        return [
            int(self.device_input)
        ]


    def read_calibration_cache(self):

        try:

            with open(
                self.cache_file,
                "rb"
            ) as f:

                return f.read()

        except FileNotFoundError:

            return None


    def write_calibration_cache(
        self,
        cache
    ):

        with open(
            self.cache_file,
            "wb"
        ) as f:

            f.write(cache)
```

---

# 10. INT8 TensorRT Engine 생성

```python
def build_int8_engine():

    builder = trt.Builder(
        TRT_LOGGER
    )

    network = builder.create_network(
        1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )
    )

    parser = trt.OnnxParser(
        network,
        TRT_LOGGER
    )

    with open(
        "mnist.onnx",
        "rb"
    ) as f:

        success = parser.parse(
            f.read()
        )

    if not success:

        for i in range(
            parser.num_errors
        ):
            print(
                parser.get_error(i)
            )

        return


    config = (
        builder.create_builder_config()
    )

    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        1 << 30
    )

    # 핵심: INT8 precision 허용
    config.set_flag(
        trt.BuilderFlag.INT8
    )


    calibrator = MNISTCalibrator(
        calibration_file="calibration_data.npy",
        batch_size=32
    )

    config.int8_calibrator = (
        calibrator
    )


    profile = (
        builder.create_optimization_profile()
    )

    profile.set_shape(
        "input",
        min=(1, 1, 28, 28),
        opt=(32, 1, 28, 28),
        max=(128, 1, 28, 28)
    )

    config.add_optimization_profile(
        profile
    )


    serialized_engine = (
        builder.build_serialized_network(
            network,
            config
        )
    )


    with open(
        "mnist_int8.engine",
        "wb"
    ) as f:

        f.write(
            serialized_engine
        )


build_int8_engine()
```

INT8에서 가장 중요한 부분은 다음 두 부분이다.

```python
config.set_flag(
    trt.BuilderFlag.INT8
)
```

그리고

```python
config.int8_calibrator = calibrator
```

이다.

첫 번째는 TensorRT에 INT8 실행을 허용하고, 두 번째는 calibration 데이터를 이용하여 quantization range를 결정하도록 한다.

---

# 11. 생성된 TensorRT Engine 크기

실제 생성된 engine의 크기는 다음과 같았다.

| Precision | Engine size |
|---|---:|
| FP32 | 901 KB |
| FP16 | 514 KB |
| INT8 | 318 KB |

FP32 대비 INT8 compression ratio는

\[
\frac{901}{318}
\approx
2.83
\]

이므로 실제 TensorRT engine 기준 약

\[
\boxed{2.83\times}
\]

작아졌다.

이론적으로 단순한 weight 저장 크기만 비교하면

\[
32\text{ bit}
\rightarrow
8\text{ bit}
\]

이므로 4배 감소를 기대할 수 있지만, 실제 `.engine` 파일에는 weight 외에도 TensorRT execution 정보, kernel 선택 정보, metadata 등이 포함되므로 정확히 4배가 되지는 않는다.

---

# 12. TensorRT Benchmark

TensorRT의 `trtexec`를 사용하여 성능을 측정하였다.

공통 설정은 다음과 같다.

```text
Warm-up: 2000 ms
Measurement duration: 10 s
Inference streams: 1
Data transfer: Enabled
```

---

## 12.1 Batch = 1

예:

```bash
/usr/src/tensorrt/bin/trtexec \
--loadEngine=mnist_fp16.engine \
--shapes=input:1x1x28x28 \
--warmUp=2000 \
--duration=10
```

INT8:

```bash
/usr/src/tensorrt/bin/trtexec \
--loadEngine=mnist_int8.engine \
--shapes=input:1x1x28x28 \
--warmUp=2000 \
--duration=10
```

측정 결과:

| Precision | Throughput | Mean latency | Mean GPU compute |
|---|---:|---:|---:|
| FP32 | 미확인 | 미확인 | 미확인 |
| FP16 | 7,781.8 qps | 0.131477 ms | 0.115445 ms |
| INT8 | 8,903.12 qps | 0.118869 ms | 0.104453 ms |

INT8은 FP16 대비 throughput 기준

\[
\frac{8903.12}{7781.8}
\approx
1.144
\]

즉 약

\[
\boxed{14.4\%}
\]

높은 throughput을 보였다.

GPU compute time을 기준으로 하면

\[
\frac{0.115445}{0.104453}
\approx
1.105
\]

이므로 약 1.10배 빨랐다.

---

# 13. Batch = 32

FP32:

```bash
/usr/src/tensorrt/bin/trtexec \
--loadEngine=mnist_fp32.engine \
--shapes=input:32x1x28x28 \
--warmUp=2000 \
--duration=10
```

FP16:

```bash
/usr/src/tensorrt/bin/trtexec \
--loadEngine=mnist_fp16.engine \
--shapes=input:32x1x28x28 \
--warmUp=2000 \
--duration=10
```

INT8:

```bash
/usr/src/tensorrt/bin/trtexec \
--loadEngine=mnist_int8.engine \
--shapes=input:32x1x28x28 \
--warmUp=2000 \
--duration=10
```

결과:

| Precision | Throughput | Mean latency | Mean GPU compute |
|---|---:|---:|---:|
| FP32 | 3,858.46 qps | 0.280964 ms | 0.254437 ms |
| FP16 | 5,635.02 qps | 0.196076 ms | 0.172999 ms |
| INT8 | 5,519.61 qps | 0.200614 ms | 0.176651 ms |

---

# 14. FP32 대비 속도 향상

## FP16

GPU compute 기준:

\[
\text{Speedup}_{FP16}
=
\frac{0.254437}
{0.172999}
\approx
1.47
\]

따라서

\[
\boxed{FP16 \approx 1.47\times}
\]

의 speedup을 보였다.

Throughput 기준:

\[
\frac{5635.02}
{3858.46}
\approx
1.46
\]

이다.

---

## INT8

GPU compute 기준:

\[
\text{Speedup}_{INT8}
=
\frac{0.254437}
{0.176651}
\approx
1.44
\]

Throughput 기준:

\[
\frac{5519.61}
{3858.46}
\approx
1.43
\]

이다.

즉 batch 32에서는

\[
\boxed{
FP16 \approx INT8 \gg FP32
}
\]

형태의 결과가 나타났다.

---

# 15. 왜 INT8이 항상 FP16보다 빠르지 않은가?

중요한 실험 결과는 batch 32에서 FP16이 INT8보다 아주 조금 빨랐다는 점이다.

```text
FP16 GPU Compute Time = 0.172999 ms
INT8 GPU Compute Time = 0.176651 ms
```

즉

\[
INT8 < FP16
\]

이라는 단순한 bit-width 관계가 곧바로

\[
T_{INT8}<T_{FP16}
\]

을 의미하지 않는다.

실제 inference 시간은 단순 arithmetic cost만으로 결정되지 않는다.

대략적으로

\[
T_{\text{inference}}
=
T_{\text{compute}}
+
T_{\text{memory}}
+
T_{\text{kernel}}
+
T_{\text{quant/dequant}}
+
T_{\text{overhead}}
\]

로 생각할 수 있다.

TensorRT는 각 layer에 대해 실제 hardware kernel 성능을 고려하여 실행 전략을 결정한다.

따라서

- 작은 CNN
- 작은 feature map
- kernel launch overhead 비중이 큰 경우
- FP16 Tensor Core kernel이 매우 효율적인 경우
- quantize/dequantize overhead가 존재하는 경우

에는 FP16이 INT8과 비슷하거나 더 빠를 수도 있다.

즉

\[
\boxed{
\text{Lower precision}
\neq
\text{Always lower latency}
}
\]

이다.

---

# 16. Batch size와 성능

Batch size가 증가하면 한 번의 inference에서 더 많은 데이터를 처리하게 된다.

예를 들어 batch 32에서는 하나의 TensorRT query가 32개의 이미지를 포함한다.

따라서 `trtexec`의 qps는 inference 호출 횟수이며, 단순 image/s와 동일하지 않다.

image throughput으로 환산하려면 대략

\[
\text{images/s}
=
\text{qps}
\times
\text{batch size}
\]

로 볼 수 있다.

예를 들어 FP16 batch 32:

\[
5635.02\times32
\approx180{,}321
\text{ images/s}
\]

이다.

다만 이 모델은 매우 작은 MNIST CNN이므로 절대적인 image/s 수치보다 **precision 간 상대 비교**가 더 의미 있다.

---

# 17. 현재까지의 최종 결과

| Precision | Accuracy | Engine size | B=1 Throughput | B=1 GPU time | B=32 Throughput | B=32 GPU time |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 98.38%* | 901 KB | 미확인 | 미확인 | 3,858.46 qps | 0.254437 ms |
| FP16 | 미측정 | 514 KB | 7,781.8 qps | 0.115445 ms | 5,635.02 qps | 0.172999 ms |
| INT8 | 미측정 | 318 KB | 8,903.12 qps | 0.104453 ms | 5,519.61 qps | 0.176651 ms |

\* FP32 accuracy 98.38%는 PyTorch 학습 모델 기준이며, TensorRT FP32 engine accuracy는 아직 별도 측정하지 않았다.

---

# 18. 현재 결과 해석

본 실험에서 관찰된 핵심 결과는 다음과 같다.

1. FP32 → FP16 → INT8로 갈수록 engine 크기가 감소하였다.
2. FP16은 FP32 대비 약 1.47배 GPU compute speedup을 보였다.
3. INT8은 FP32 대비 약 1.44배 GPU compute speedup을 보였다.
4. Batch=1에서는 INT8이 FP16보다 빨랐다.
5. Batch=32에서는 FP16이 INT8보다 약간 빨랐다.
6. 따라서 정밀도를 낮춘다고 항상 latency가 동일한 비율로 감소하지는 않는다.
7. 실제 Edge AI에서는 accuracy, latency, throughput, memory/storage를 동시에 고려해야 한다.

이를 multi-objective 관점에서 보면 모델 선택 문제는 다음과 같이 생각할 수 있다.

\[
\max
\{
\text{Accuracy},
\text{Throughput}
\}
\]

동시에

\[
\min
\{
\text{Latency},
\text{Model Size},
\text{Memory}
\}
\]

를 만족하는 precision을 찾는 문제이다.

따라서 FP32, FP16, INT8 중 하나가 모든 지표에서 항상 최적이라고 볼 수 없고, 실제 application 요구사항에 따라 Pareto-optimal한 선택이 달라질 수 있다.

---

# 19. 다음 실험

현재 `trtexec` benchmark는 random input을 사용하여 속도를 측정하였다.

따라서 다음 단계는 저장해둔

```text
test_data.npy
test_labels.npy
```

를 실제 FP32, FP16, INT8 TensorRT engine에 입력하여 다음 정확도를 비교하는 것이다.

\[
Acc_{FP32}
\]

\[
Acc_{FP16}
\]

\[
Acc_{INT8}
\]

최종적으로 다음 세 지표를 함께 비교한다.

```text
Accuracy
   vs
Latency / Throughput
   vs
Engine Size
```

이를 통해 quantization의 실제 trade-off를 분석할 수 있다.
