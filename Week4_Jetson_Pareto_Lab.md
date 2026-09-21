# Week 4: Colab에서 학습한 모델을 Jetson에서 추론하고 Pareto Front 분석하기

## 1. 실습 목표

이번 실습의 목적은 Jetson에서 모델을 학습하는 것이 아니라, **PC 또는 Google Colab에서 학습한 모델을 Jetson에 배포하고 실제 AIoT 하드웨어의 추론 성능을 측정하는 것**입니다.

```text
학생 PC / Google Colab
        ↓
MNIST 모델 학습
        ↓
PyTorch 가중치(.pth) 저장
        ↓
Jetson으로 전송
        ↓
GPU/CUDA 동작 확인
        ↓
학습된 모델로 추론
        ↓
Accuracy와 Latency 측정
        ↓
Pareto Front 분석
```

핵심 개념은 다음과 같습니다.

> 모델을 학습하는 장치와 실제로 사용하는 장치는 다를 수 있다.

| 단계 | 실행 환경 | 주요 작업 |
|---|---|---|
| Training | PC, Colab 또는 GPU 서버 | 모델 학습 및 가중치 생성 |
| Deployment | Jetson | 모델과 가중치 배치 |
| Inference | Jetson | 정확도와 지연 시간 측정 |
| Decision | Jetson / 분석 환경 | Accuracy–Latency Pareto Front 분석 |

---

## 2. 전체 실습 순서

이번 실습은 다음 순서로 진행합니다.

```text
Part A. Colab에서 TinyMLP 학습
        ↓
Part B. TinyMLP.pth 저장 및 Jetson 전송
        ↓
Part C. Jetson 환경 설정
        ↓
Part D. check_gpu.py
        ↓
Part E. latency_toy.py
        ↓
Part F. pareto_check.py
        ↓
Part G. 결과 해석 및 토론
```

이번 시간에는 Pareto 분석 코드의 동작 확인을 위해 **TinyMLP만 실제 측정**하고, 나머지 모델은 예시 값을 사용합니다.

> [!IMPORTANT]
> `SmallCNN`, `MediumCNN`, `LargeCNN`의 Accuracy/Latency 값은 Pareto Front 실습을 위한 **가상 예시값**입니다. 실제 실험 결과가 아닙니다.

---

# Part A. Colab에서 TinyMLP 학습

## 3. TinyMLP 구조

TinyMLP는 MNIST의 28×28 흑백 이미지를 입력받아 10개의 숫자 클래스를 분류하는 간단한 다층 퍼셉트론입니다.

```text
입력 이미지
1 × 28 × 28
    ↓
Flatten
784
    ↓
Linear(784, 32)
    ↓
ReLU
    ↓
Linear(32, 10)
    ↓
숫자 클래스 0~9
```

---

## 4. Google Colab에서 TinyMLP 학습

아래 코드를 Colab의 하나의 셀에 붙여 넣고 실행합니다.

```python
# ============================================================
# TinyMLP MNIST Training for Google Colab
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

from google.colab import files
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. Device and Hyperparameters
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 5

print("Device:", device)

if device.type == "cuda":
    print("GPU   :", torch.cuda.get_device_name(0))


# ============================================================
# 2. MNIST Dataset
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,),
    ),
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# 3. Model Definition
# ============================================================

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)


model = TinyMLP().to(device)

print()
print(model)


# ============================================================
# 4. Loss and Optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 5. Training
# ============================================================

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
        f" | Loss = {avg_loss:.4f}"
    )


# ============================================================
# 6. Accuracy Check
# ============================================================

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

accuracy = correct / total * 100

print()
print(f"Test Accuracy: {accuracy:.2f}%")


# ============================================================
# 7. Save Weights
# ============================================================

torch.save(
    model.state_dict(),
    "TinyMLP.pth",
)

print()
print("Saved: TinyMLP.pth")


# ============================================================
# 8. Download
# ============================================================

files.download("TinyMLP.pth")
```

정상 실행 시 다음과 같은 결과를 확인할 수 있습니다.

```text
Epoch 1/5 | Loss = ...
Epoch 2/5 | Loss = ...
...
Test Accuracy: ...%
Saved: TinyMLP.pth
```

`TinyMLP.pth`에는 전체 모델 객체가 아니라 `model.state_dict()`, 즉 **학습된 파라미터만 저장**됩니다.

---

# Part B. 학습된 모델을 Jetson으로 전송

## 5. 가중치 파일 배치

다운로드한 `TinyMLP.pth`를 Jetson의 프로젝트 디렉터리에 복사합니다.

최종 프로젝트 구조는 다음과 같이 구성합니다.

```text
week4_performance/
├─ week4_env/
├─ check_gpu.py
├─ latency_toy.py
├─ pareto_check.py
│
└─ weights/
   └─ TinyMLP.pth
```

Jetson에서 파일을 확인합니다.

```bash
ls -lh weights/TinyMLP.pth
```

---

# Part C. Jetson 환경 설정

## 6. 실습 환경

본 실습은 다음 환경을 기준으로 합니다.

```text
Jetson Orin Nano
Ubuntu 22.04
L4T R36.4.7
CUDA 12.6
Python 3.10
PyTorch 2.8.0
torchvision 0.23.0
NumPy 1.26.4
```

---

## 7. 가상환경 생성

```bash
mkdir -p ~/week4_performance
cd ~/week4_performance

python3.10 -m venv week4_env
source week4_env/bin/activate

python -m pip install --upgrade pip
```

가상환경이 활성화되면 터미널 앞에 다음과 같이 표시됩니다.

```text
(week4_env)
```

---

## 8. NumPy 버전 고정

Jetson용 PyTorch/torchvision wheel과의 ABI 호환성을 위해 NumPy를 1.x 버전으로 고정합니다.

```bash
pip install numpy==1.26.4
```

확인:

```bash
python -c "import numpy; print(numpy.__version__)"
```

정상 출력:

```text
1.26.4
```

---

## 9. cuSPARSELt 설치

PyTorch가 필요로 하는 Jetson CUDA 의존성을 먼저 설치합니다.

```bash
wget https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb

sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb

sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.8.1/cusparselt-local-tegra-C4CC87E1-keyring.gpg \
    /usr/share/keyrings/

sudo apt-get update
sudo apt-get -y install cusparselt-cuda-12
```

---

## 10. Jetson용 PyTorch와 torchvision 설치

> [!IMPORTANT]
> 일반적인 `pip install torch torchvision`을 사용하지 않습니다.  
> Jetson의 aarch64 / CUDA 12.6 환경에 맞는 wheel을 사용합니다.

교수자가 다음 두 wheel 파일을 사전에 제공합니다.

```text
torch-2.8.0-cp310-cp310-linux_aarch64.whl
torchvision-0.23.0-cp310-cp310-linux_aarch64.whl
```

설치:

```bash
pip install torch-2.8.0-cp310-cp310-linux_aarch64.whl

pip install torchvision-0.23.0-cp310-cp310-linux_aarch64.whl --no-deps
```

기타 실습 패키지 설치:

```bash
pip install pandas matplotlib
```

---

# Part D. GPU/CUDA 동작 확인

## 11. `check_gpu.py`

단순히 `torch.cuda.is_available()`만 확인하지 않고, 실제 CUDA 텐서의 행렬 곱셈까지 실행하여 GPU와 cuBLAS가 정상 동작하는지 확인합니다.

```python
# ============================================================
# check_gpu.py
# Week 4 - GPU / CUDA Check
# ============================================================

import sys
import torch


print("=" * 60)
print("GPU / CUDA Environment")
print("=" * 60)

print("Python version :", sys.version.split()[0])
print("PyTorch version:", torch.__version__)
print("CUDA version   :", torch.version.cuda)
print("CUDA available :", torch.cuda.is_available())

print()

if not torch.cuda.is_available():
    print("Device         : CPU")
    raise SystemExit(
        "CUDA is not available. Check the Jetson environment."
    )

print("Device         : CUDA")
print("GPU            :", torch.cuda.get_device_name(0))
print("GPU count      :", torch.cuda.device_count())


# ============================================================
# Simple CUDA / cuBLAS Test
# ============================================================

x = torch.randn(
    100,
    100,
    device="cuda",
)

y = x @ x

torch.cuda.synchronize()

print()
print("Matrix multiplication test")
print("Result device  :", y.device)
print("Result shape   :", y.shape)
print("CUDA test      : SUCCESS")

print("=" * 60)
```

실행:

```bash
python check_gpu.py
```

정상적인 예:

```text
CUDA available : True
Device         : CUDA
GPU            : Orin

Matrix multiplication test
Result device  : cuda:0
Result shape   : torch.Size([100, 100])
CUDA test      : SUCCESS
```

---

# Part E. Latency 측정 원리 이해

## 12. 추론 지연 시간 측정 시 고려사항

이번 실습에서는 **single-image inference latency**를 측정합니다.

### 12.1 여러 번 반복 측정

한 번의 측정값은 운영체제 스케줄링, GPU 상태, 시스템 부하 등에 영향을 받을 수 있습니다.

따라서 여러 번 반복하여 다음 값을 확인합니다.

- Mean
- Median
- Standard deviation

### 12.2 Warm-up

처음 몇 번의 실행에는 CUDA context 생성 및 초기 커널 준비 비용이 포함될 수 있습니다.

따라서 실제 측정 전에 여러 번 추론을 수행합니다.

### 12.3 CUDA synchronization

CUDA 연산은 기본적으로 비동기(asynchronous)로 실행됩니다.

따라서 GPU 연산 시간을 정확히 측정하려면 시간 측정 전후에 다음 함수를 사용합니다.

```python
torch.cuda.synchronize()
```

---

## 13. `latency_toy.py`

학습 여부와 관계없는 작은 CNN을 이용해 latency 측정 방법 자체를 확인합니다.

```python
# ============================================================
# latency_toy.py
# Week 4 - Toy Inference Latency Measurement
# ============================================================

import time
import numpy as np

import torch
import torch.nn as nn


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("Latency Measurement")
print("=" * 60)

print("Device :", device)

if device.type == "cuda":
    print("GPU    :", torch.cuda.get_device_name(0))

print()


# ============================================================
# 2. Toy CNN
# ============================================================

class ToyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(
                16 * 14 * 14,
                10,
            ),
        )

    def forward(self, x):
        return self.model(x)


model = ToyCNN().to(device)
model.eval()


# ============================================================
# 3. Dummy Input
# ============================================================

# Single-image inference
x = torch.randn(
    1,
    1,
    28,
    28,
    device=device,
)


# ============================================================
# 4. Warm-up
# ============================================================

WARMUP = 50

with torch.no_grad():
    for _ in range(WARMUP):
        _ = model(x)

if device.type == "cuda":
    torch.cuda.synchronize()


# ============================================================
# 5. Latency Measurement
# ============================================================

REPEAT = 200
latencies = []

with torch.no_grad():
    for _ in range(REPEAT):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        latencies.append(latency_ms)


# ============================================================
# 6. Results
# ============================================================

mean_latency = np.mean(latencies)
median_latency = np.median(latencies)
std_latency = np.std(latencies)

print("=" * 60)
print("Single-image Inference Latency")
print("=" * 60)

print(f"Mean   : {mean_latency:.4f} ms")
print(f"Median : {median_latency:.4f} ms")
print(f"Std    : {std_latency:.4f} ms")

print("=" * 60)
```

실행:

```bash
python latency_toy.py
```

이 실습의 목적은 모델 성능을 비교하는 것이 아니라 다음 세 가지를 이해하는 것입니다.

```text
Warm-up
    ↓
반복 추론
    ↓
CUDA synchronization
    ↓
Latency 통계 계산
```

---

# Part F. 실제 모델 추론 및 Pareto Front

## 14. Colab과 Jetson의 모델 구조 일치

Jetson에서 `.pth` 파일을 불러올 때는 Colab에서 사용한 모델 구조와 **완전히 동일한 클래스**가 필요합니다.

TinyMLP의 핵심 구조는 다음과 같습니다.

```python
nn.Linear(28 * 28, 32)
nn.Linear(32, 10)
```

예를 들어 Jetson에서 다음과 같이 바꾸면 안 됩니다.

```python
nn.Linear(28 * 28, 64)
```

이 경우 저장된 가중치의 shape이 다르므로 다음과 같은 오류가 발생합니다.

```text
size mismatch for ...
```

또한 각 모델에는 해당 모델에서 학습한 weight를 사용해야 합니다.

```text
TinyMLP.pth  → TinyMLP
SmallCNN.pth → SmallCNN
```

다른 구조의 모델에 `TinyMLP.pth`를 사용할 수 없습니다.

---

## 15. 학습과 추론의 전처리 일치

Colab 학습 시 사용한 전처리와 Jetson 평가 시 사용하는 전처리는 같아야 합니다.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,),
    ),
])
```

전처리가 다르면 같은 weight를 사용하더라도 accuracy가 달라질 수 있습니다.

---

## 16. Pareto Front의 목적함수

이번 실습에서는 두 개의 목적을 동시에 고려합니다.

| 지표 | 최적화 방향 |
|---|---|
| Accuracy | 높을수록 좋음 |
| Latency | 낮을수록 좋음 |

모델 A가 모델 B에 비해

- Accuracy가 같거나 높고,
- Latency가 같거나 낮으며,
- 둘 중 적어도 하나에서 더 좋다면

A가 B를 **dominate**한다고 합니다.

예를 들어:

```text
Model A
Accuracy = 98.0%
Latency  = 0.5 ms

Model B
Accuracy = 97.0%
Latency  = 0.8 ms
```

Model A는 Model B보다 정확하면서 더 빠르므로 Model B는 dominated solution입니다.

반대로:

```text
Model C
Accuracy = 99.0%
Latency  = 1.5 ms
```

는 A보다 정확하지만 더 느립니다.

이 경우 A와 C 사이에는 trade-off가 존재하며 둘 다 Pareto Front에 포함될 수 있습니다.

---

## 17. `pareto_check.py`

이번 버전에서는 **TinyMLP만 실제 Jetson에서 측정**합니다.

다른 모델의 값은 Pareto 분석 코드 동작을 확인하기 위한 예시값입니다.

```python
# ============================================================
# pareto_check.py
#
# Week 4 - Accuracy / Latency Pareto Analysis
#
# Actual measurement:
#   TinyMLP
#
# Demo values:
#   SmallCNN / MediumCNN / LargeCNN / EfficientCNN
# ============================================================

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. Settings
# ============================================================

BATCH_SIZE = 128

LATENCY_WARMUP = 50
LATENCY_REPEAT = 200


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("Device :", device)

if device.type == "cuda":
    print(
        "GPU    :",
        torch.cuda.get_device_name(0),
    )

print("=" * 60)


# ============================================================
# 3. MNIST Test Dataset
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,),
    ),
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# 4. TinyMLP
# ============================================================

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 5. Accuracy
# ============================================================

def measure_accuracy(model):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        correct
        / total
        * 100
    )


# ============================================================
# 6. Latency
# ============================================================

def measure_latency(model):
    model.eval()

    # Single-image input
    x = torch.randn(
        1,
        1,
        28,
        28,
        device=device,
    )

    # Warm-up
    with torch.no_grad():
        for _ in range(
            LATENCY_WARMUP
        ):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []

    with torch.no_grad():
        for _ in range(
            LATENCY_REPEAT
        ):

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            latencies.append(
                (end - start)
                * 1000
            )

    return np.mean(latencies)


# ============================================================
# 7. Load Actual Model
# ============================================================

model = TinyMLP()

state_dict = torch.load(
    "weights/TinyMLP.pth",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    state_dict
)

model = model.to(device)
model.eval()


# ============================================================
# 8. Actual Measurement
# ============================================================

print()
print("=" * 60)
print("TinyMLP - Actual Measurement")
print("=" * 60)

tiny_accuracy = measure_accuracy(
    model
)

tiny_latency = measure_latency(
    model
)

print(
    f"Accuracy : "
    f"{tiny_accuracy:.2f} %"
)

print(
    f"Latency  : "
    f"{tiny_latency:.4f} ms"
)


results = [
    {
        "Model": "TinyMLP",
        "Accuracy": tiny_accuracy,
        "Latency": tiny_latency,
        "Source": "Measured",
    }
]


# ============================================================
# 9. Demo Results
#
# These values are NOT real measurements.
# They are used only to demonstrate Pareto analysis.
# ============================================================

fake_results = [
    {
        "Model": "EfficientCNN",
        "Accuracy": 96.8,
        "Latency": 0.42,
        "Source": "Demo",
    },
    {
        "Model": "SmallCNN",
        "Accuracy": 97.8,
        "Latency": 0.62,
        "Source": "Demo",
    },
    {
        "Model": "MediumCNN",
        "Accuracy": 97.2,
        "Latency": 0.95,
        "Source": "Demo",
    },
    {
        "Model": "LargeCNN",
        "Accuracy": 99.0,
        "Latency": 1.65,
        "Source": "Demo",
    },
]

results.extend(
    fake_results
)


# ============================================================
# 10. DataFrame
# ============================================================

df = pd.DataFrame(
    results
)


# ============================================================
# 11. Pareto Dominance
# ============================================================

def is_dominated(
    i,
    dataframe,
):

    acc_i = dataframe.loc[
        i,
        "Accuracy"
    ]

    latency_i = dataframe.loc[
        i,
        "Latency"
    ]

    for j in dataframe.index:

        if i == j:
            continue

        acc_j = dataframe.loc[
            j,
            "Accuracy"
        ]

        latency_j = dataframe.loc[
            j,
            "Latency"
        ]

        no_worse = (
            acc_j >= acc_i
            and
            latency_j <= latency_i
        )

        strictly_better = (
            acc_j > acc_i
            or
            latency_j < latency_i
        )

        if (
            no_worse
            and
            strictly_better
        ):
            return True

    return False


df["Pareto"] = True

for i in df.index:

    if is_dominated(
        i,
        df,
    ):
        df.loc[
            i,
            "Pareto"
        ] = False


# ============================================================
# 12. Print Results
# ============================================================

print()
print("=" * 60)
print("Accuracy / Latency Results")
print("=" * 60)

print(
    df.to_string(
        index=False
    )
)


# ============================================================
# 13. Pareto Front
# ============================================================

pareto_df = df[
    df["Pareto"]
].copy()

pareto_df = pareto_df.sort_values(
    "Latency"
)


# ============================================================
# 14. Plot
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["Latency"],
    df["Accuracy"],
    s=80,
)

for _, row in df.iterrows():

    label = (
        f"{row['Model']}"
        f" ({row['Source']})"
    )

    plt.annotate(
        label,
        (
            row["Latency"],
            row["Accuracy"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.plot(
    pareto_df["Latency"],
    pareto_df["Accuracy"],
    marker="o",
    label="Pareto Front",
)

plt.xlabel(
    "Single-image Latency (ms)"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Accuracy vs. Latency"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "pareto_accuracy_latency.png",
    dpi=200,
)

df.to_csv(
    "pareto_results.csv",
    index=False,
)

print()
print(
    "Saved: pareto_results.csv"
)

print(
    "Saved: pareto_accuracy_latency.png"
)
```

실행:

```bash
python pareto_check.py
```

---

# Part G. 결과 해석 및 토론

## 18. 결과 해석

학생들은 생성된 다음 파일을 확인합니다.

```text
pareto_results.csv
pareto_accuracy_latency.png
```

예시 결과는 다음과 같은 형태입니다.

| Model | Accuracy | Latency | Source | Pareto |
|---|---:|---:|---|---|
| TinyMLP | 실제 측정값 | 실제 측정값 | Measured | True/False |
| EfficientCNN | 96.8 | 0.42 | Demo | True/False |
| SmallCNN | 97.8 | 0.62 | Demo | True/False |
| MediumCNN | 97.2 | 0.95 | Demo | False 예상 |
| LargeCNN | 99.0 | 1.65 | Demo | True/False |

`MediumCNN`은 `SmallCNN`에 비해

- Accuracy가 낮고
- Latency가 높기 때문에

지배당하는(dominated) 모델이 되도록 예시값을 구성했습니다.

---

## 19. 토론 질문

1. 가장 정확한 모델이 항상 배포에 가장 적합한가?
2. 가장 빠른 모델이 항상 좋은 모델인가?
3. 실시간 AIoT 장치에서는 Accuracy와 Latency 중 어느 쪽을 더 중요하게 봐야 하는가?
4. 어떤 모델이 Pareto Front에 포함되는가?
5. dominated model을 실제 배포 후보에서 제외해도 되는가?
6. 허용 latency가 1 ms 이하라면 어떤 모델을 선택할 수 있는가?
7. Accuracy가 최소 97% 이상이어야 한다면 어떤 후보가 남는가?

---

# Part H. 다음 확장 실습

## 20. FLOPs / MACs 추가

다음 단계에서는 FLOPs 또는 MACs를 측정하여 이론적 연산량과 실제 하드웨어 성능의 관계를 비교합니다.

```text
Training
    ↓
Deployment
    ↓
Accuracy       → 모델 품질
Latency        → 실제 하드웨어 성능
FLOPs / MACs   → 이론적 연산 복잡도
```

핵심 질문:

> FLOPs가 적은 모델은 Jetson에서도 반드시 latency가 짧을까?

반드시 그렇지는 않습니다.

실제 latency에는 다음 요소도 영향을 줍니다.

- Memory access
- Kernel launch overhead
- Parallelism
- GPU utilization
- Operator 종류
- Hardware-specific optimization

따라서 FLOPs/MACs와 실제 latency를 함께 측정하는 것이 중요합니다.

---

# Appendix A. NumPy ABI 충돌

## A.1 증상

다음과 같은 메시지가 나타날 수 있습니다.

```text
A module that was compiled using NumPy 1.x
cannot be run in NumPy 2.x
```

또는:

```text
RuntimeError: Numpy is not available
```

## A.2 해결

현재 NumPy 제거:

```bash
pip uninstall numpy -y
```

NumPy 1.26.4 설치:

```bash
pip install numpy==1.26.4
```

확인:

```bash
python -c "import numpy; print(numpy.__version__)"
```

PyTorch와 NumPy 연동 확인:

```bash
python -c "import torch; x=torch.tensor([1,2,3]); print(x.numpy())"
```

정상 출력:

```text
[1 2 3]
```

---

# Appendix B. 자주 발생하는 오류

## B.1 `size mismatch for ...`

원인:

- `.pth`를 생성한 모델 구조와 현재 모델 구조가 다름
- 다른 모델용 weight 파일을 잘못 사용함

예:

```text
TinyMLP.pth → TinyMLP         O
TinyMLP.pth → LargeMLP        X
TinyMLP.pth → SmallCNN        X
```

해결:

- Colab과 Jetson 모델 클래스를 동일하게 작성
- 각 모델에 맞는 `.pth` 파일 사용

---

## B.2 CUDA는 True인데 행렬 곱셈이 실패함

다음 테스트를 수행합니다.

```bash
python -c "import torch; x=torch.randn(100,100,device='cuda'); y=x@x; print(y.shape, y.device)"
```

정상 출력:

```text
torch.Size([100, 100]) cuda:0
```

이 테스트가 실패하면 Python 코드보다 Jetson의 CUDA/PyTorch/cuBLAS 환경을 먼저 확인해야 합니다.

---

## B.3 `CUDA available : False`

확인:

```bash
nvcc --version
```

```bash
nvidia-smi
```

그리고 Jetson 환경에 맞는 PyTorch wheel이 설치되었는지 확인합니다.

---

# 빠른 점검표

- [ ] Colab에서 TinyMLP 학습
- [ ] `TinyMLP.pth` 다운로드
- [ ] Jetson `weights/` 디렉터리로 파일 전송
- [ ] `week4_env` 활성화
- [ ] NumPy 1.26.4 확인
- [ ] Jetson용 PyTorch / torchvision 설치 확인
- [ ] `python check_gpu.py` 성공
- [ ] CUDA matrix multiplication 성공
- [ ] `python latency_toy.py` 성공
- [ ] TinyMLP weight 로딩 성공
- [ ] TinyMLP Accuracy 측정
- [ ] TinyMLP single-image latency 측정
- [ ] Demo 모델과 함께 Pareto 분석
- [ ] `pareto_results.csv` 생성 확인
- [ ] `pareto_accuracy_latency.png` 생성 확인
- [ ] Pareto Front 해석 및 토론
