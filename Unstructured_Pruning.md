# Week 5: Jetson Unstructured Pruning 실습

## 1. 실습 목표

이 실습에서는 Jetson Orin Nano에서 MNIST 분류용 SmallCNN을 직접 학습한 뒤, global unstructured magnitude pruning을 적용합니다.

Pruning 비율을 0%, 30%, 50%, 70%, 90%로 변경하면서 다음 항목을 비교합니다.

- 분류 정확도
- 실제 weight sparsity
- Single-image inference latency
- 모델 파라미터 수
- 저장 파일 크기

핵심 질문은 다음과 같습니다.

> Weight의 90%를 0으로 만들면 Jetson의 추론 속도도 10배 빨라질까?


> [!NOTE]
> 본 문서는 **Unstructured Pruning 실습 전용**입니다. Structured pruning과 fine-tuning은 별도 실습에서 다룹니다.

```text
환경 설정
    ↓
GPU/CUDA 확인
    ↓
MNIST 다운로드
    ↓
SmallCNN 학습
    ↓
Baseline 가중치 저장
    ↓
Global unstructured pruning
    ↓
Accuracy / Sparsity / Latency 측정
    ↓
결과 해석
```


## 2. 실습 환경

이 문서는 다음 환경을 기준으로 작성했습니다.

```text
Device     : Jetson Orin Nano
OS         : Ubuntu 22.04
CUDA       : 12.6
Python     : 3.10
PyTorch    : Jetson용 NVIDIA build
GPU CC     : 8.7 (sm_87)
```

> [!IMPORTANT]
> Jetson에서는 일반 PC용 PyTorch wheel을 설치하지 않습니다. 반드시 현재 장비의 JetPack/L4T 및 CUDA 버전에 맞는 Jetson용 NVIDIA build를 사용합니다.
> 본 실습은 CUDA 12.6, Python 3.10, Orin(sm_87) 환경에서 검증했습니다.

## 3. 작업 디렉터리와 가상환경 생성

Jetson 터미널에서 다음 명령을 실행합니다.

```bash
mkdir -p ~/unstructured_pruning_lab
cd ~/unstructured_pruning_lab

python3.10 -m venv pruning_env
source pruning_env/bin/activate

python -m pip install --upgrade pip
```

터미널 프롬프트 앞에 `(pruning_env)`가 표시되는지 확인합니다.

이후 터미널을 새로 열었다면 다음 명령으로 다시 실습 환경에 들어갑니다.

```bash
cd ~/unstructured_pruning_lab
source pruning_env/bin/activate
```

## 4. cuSPARSELt 설치

일부 Jetson용 PyTorch wheel이 의존하는 cuSPARSELt 패키지를 설치합니다. 이 패키지는 본 unstructured pruning 알고리즘 자체의 필수 구성요소가 아니라, Jetson용 PyTorch 실행 환경을 맞추기 위한 의존성입니다.

```bash
wget https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb
```

```bash
sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb
```

```bash
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.8.1/cusparselt-local-tegra-C4CC87E1-keyring.gpg /usr/share/keyrings/
```

```bash
sudo apt-get update
sudo apt-get -y install cusparselt-cuda-12
```

## 5. Jetson용 PyTorch 환경 설치

### 5.1 Jetson용 PyTorch 준비

Jetson에서는 장비의 JetPack/L4T에 맞는 NVIDIA 제공 PyTorch wheel을 사용합니다.
일반 `pip install torch`는 CUDA 또는 GPU compute capability가 맞지 않을 수 있으므로 사용하지 않습니다.

설치 전 현재 장비 정보를 확인합니다.

```bash
cat /etc/nv_tegra_release
nvcc --version
python3 --version
```

교수자가 제공한 Jetson용 PyTorch wheel 또는 현재 장비에 맞는 NVIDIA 공식 wheel을 준비합니다.

### 5.2 NumPy 버전 고정

Jetson용 wheel과 NumPy 2.x 사이의 ABI 충돌을 방지하기 위해 NumPy를 1.26.4로 고정합니다.

```bash
pip install numpy==1.26.4
```

> [!WARNING]
> 이 환경에서 `pip install numpy`만 실행하면 최신 NumPy 2.x가 설치될 수 있습니다. 반드시 버전을 명시하세요.

### 5.3 PyTorch와 실습 패키지 설치

아래 `<JETSON_TORCH_WHEEL>`은 교수자가 제공하거나 장비에 맞게 준비한 Jetson용 wheel 파일명으로 바꿉니다.

```bash
pip install ./<JETSON_TORCH_WHEEL>
pip install pandas matplotlib
```

`torchvision`은 이 실습의 필수 패키지가 아닙니다. 설치가 필요한 경우에는 반드시 현재 Jetson용 PyTorch와 호환되는 버전을 사용합니다.

### 5.4 설치 버전 확인

```bash
python -c "import numpy, torch; print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Arch list:', torch.cuda.get_arch_list() if torch.cuda.is_available() else None)"
```

Orin에서는 `sm_87` 또는 `compute_87`이 지원 목록에 포함되어 있는지 확인합니다.

NumPy는 다음 버전이어야 합니다.

```text
NumPy: 1.26.4
```

PyTorch와 NumPy의 연동도 확인합니다.

```bash
python -c "import torch; x = torch.tensor([1, 2, 3]); print(x.numpy())"
```

정상 출력:

```text
[1 2 3]
```

## 6. CUDA 동작 확인

`check_gpu.py`를 생성합니다.

```bash
cat > check_gpu.py <<'PY'
import sys
import torch


print("=" * 60)
print("GPU / CUDA Environment")
print("=" * 60)
print("Python version :", sys.version.split()[0])
print("PyTorch version:", torch.__version__)
print("CUDA version   :", torch.version.cuda)
print("CUDA available :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Stop the lab and check PyTorch installation.")

print("GPU            :", torch.cuda.get_device_name(0))
print("GPU count      :", torch.cuda.device_count())

x = torch.randn(100, 100, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("Result device  :", y.device)
print("Result shape   :", tuple(y.shape))
print("CUDA test      : SUCCESS")
PY
```

실행합니다.

```bash
python check_gpu.py
```

마지막에 다음 메시지가 출력되어야 합니다.

```text
CUDA available : True
Result device  : cuda:0
CUDA test      : SUCCESS
```

CUDA 테스트가 실패하면 이후 latency 결과가 GPU 측정값이 아니므로 다음 단계로 진행하지 않습니다.

## 7. SmallCNN 모델 정의

`model.py`를 생성합니다.

```bash
cat > model.py <<'PY'
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
PY
```

생성된 파일을 확인합니다.

```bash
cat model.py
```

## 8. Baseline 모델 학습

`train_baseline.py`는 다음 작업을 한 번에 수행합니다.

1. MNIST gzip 파일을 `data/`에서 직접 읽기
2. 입력을 `[0,1]` 범위로 정규화
3. SmallCNN 학습
4. 시험 데이터 정확도 측정
5. Baseline 가중치 저장
6. Pruning 평가용 시험 데이터 저장

> [!IMPORTANT]
> 이 문서에서는 `torchvision`에 의존하지 않습니다. 학습과 평가에서 동일하게 `x / 255.0` 전처리를 사용하여 입력 분포를 일관되게 유지합니다.

먼저 MNIST 원본 gzip 파일이 `data/`에 있어야 합니다.

```text
data/
├─ train-images-idx3-ubyte.gz
├─ train-labels-idx1-ubyte.gz
├─ t10k-images-idx3-ubyte.gz
└─ t10k-labels-idx1-ubyte.gz
```

파일을 생성합니다.

```bash
cat > train_baseline.py <<'PY'
import gzip
import random
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from model import SmallCNN


# ============================================================
# 1. Reproducibility and Device
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type != "cuda":
    raise RuntimeError("CUDA is not available.")

print("Device:", device)
print("GPU   :", torch.cuda.get_device_name(0))


# ============================================================
# 2. Hyperparameters
# ============================================================

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3


# ============================================================
# 3. MNIST Loader
# ============================================================

def load_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n, 1, rows, cols)

    return data.astype(np.float32) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels.astype(np.int64)


data_dir = Path("./data")

train_images = load_images(data_dir / "train-images-idx3-ubyte.gz")
train_labels = load_labels(data_dir / "train-labels-idx1-ubyte.gz")

test_images = load_images(data_dir / "t10k-images-idx3-ubyte.gz")
test_labels = load_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

print("Train:", train_images.shape)
print("Test :", test_images.shape)


train_dataset = TensorDataset(
    torch.tensor(train_images, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.long),
)

test_dataset = TensorDataset(
    torch.tensor(test_images, dtype=torch.float32),
    torch.tensor(test_labels, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# 4. Model, Loss, and Optimizer
# ============================================================

model = SmallCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


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

    average_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {average_loss:.4f}")


# ============================================================
# 6. Baseline Accuracy
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

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100.0 * correct / total
print(f"Baseline Accuracy: {accuracy:.2f}%")


# ============================================================
# 7. Save Weights and Test Data
# ============================================================

torch.save(model.state_dict(), "mnist_fp32.pth")
np.save("test_data.npy", test_images)
np.save("test_labels.npy", test_labels)

print("Saved: mnist_fp32.pth")
print("Saved: test_data.npy", test_images.shape)
print("Saved: test_labels.npy", test_labels.shape)
PY
```

학습을 실행합니다.

```bash
python train_baseline.py
```

완료되면 다음과 같은 결과가 출력됩니다.

```text
Device: cuda
GPU   : Orin
Train: (60000, 1, 28, 28)
Test : (10000, 1, 28, 28)
Epoch 1/5 | Loss: ...
...
Baseline Accuracy: ...%
Saved: mnist_fp32.pth
Saved: test_data.npy (10000, 1, 28, 28)
Saved: test_labels.npy (10000,)
```

생성된 파일을 확인합니다.

```bash
ls -lh mnist_fp32.pth test_data.npy test_labels.npy
```

## 9. Unstructured Pruning 원리

Magnitude pruning은 절댓값이 작은 weight를 중요도가 낮다고 보고 0으로 만듭니다.

$$
\operatorname{importance}(w_i)=|w_i|
$$

이번 실습에서는 네 개 계층의 weight를 하나의 집합으로 보고 global pruning을 적용합니다.

```text
Conv1 weight ┐
Conv2 weight ├─ 전체 weight의 |w| 비교 ─→ 작은 값부터 제거
FC1 weight   │
FC2 weight   ┘
```

Global pruning에서는 전체 sparsity가 50%여도 각 계층의 sparsity가 정확히 50%가 되지는 않습니다. 전체 계층을 합친 뒤 절댓값이 가장 작은 weight 50%를 제거하기 때문입니다.

또한 unstructured pruning은 weight 값을 0으로 만들 뿐 tensor shape을 줄이지 않습니다.

```text
Pruning 전: Linear weight shape = 128 × 1568
Pruning 후: Linear weight shape = 128 × 1568
                              └─ 일부 원소만 0
```

## 10. Pruning Sweep 실험

0%, 30%, 50%, 70%, 90% pruning을 한 번에 실행하는 `pruning_sweep.py`를 생성합니다.

```bash
cat > pruning_sweep.py <<'PY'
import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.utils.prune as prune

from model import SmallCNN


# ============================================================
# 1. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type != "cuda":
    raise RuntimeError("CUDA is not available.")

print("Device:", device)
print("GPU   :", torch.cuda.get_device_name(0))


# ============================================================
# 2. Load Test Data
# ============================================================

x_test = torch.tensor(
    np.load("test_data.npy"),
    dtype=torch.float32,
)

y_test = torch.tensor(
    np.load("test_labels.npy"),
    dtype=torch.long,
)

print("X shape:", tuple(x_test.shape))
print("Y shape:", tuple(y_test.shape))


# ============================================================
# 3. Helper Functions
# ============================================================

def pruning_targets(model):
    return (
        (model.features[0], "weight"),
        (model.features[3], "weight"),
        (model.classifier[1], "weight"),
        (model.classifier[3], "weight"),
    )


def evaluate_accuracy(model):
    model.eval()
    correct = 0
    total = 0
    batch_size = 256

    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            images = x_test[start:start + batch_size].to(device)
            labels = y_test[start:start + batch_size].to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def measure_sparsity(model):
    zero_count = 0
    weight_count = 0

    for module, _ in pruning_targets(model):
        zero_count += (module.weight == 0).sum().item()
        weight_count += module.weight.numel()

    return 100.0 * zero_count / weight_count


def measure_latency(model, warmup=100, repeat=500):
    model.eval()
    sample = x_test[:1].to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)

    torch.cuda.synchronize()
    latencies = []

    with torch.no_grad():
        for _ in range(repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(sample)

            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

    return {
        "mean": float(np.mean(latencies)),
        "median": float(np.median(latencies)),
        "std": float(np.std(latencies)),
    }


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    nonzero = sum(torch.count_nonzero(parameter).item() for parameter in model.parameters())
    return total, nonzero


# ============================================================
# 4. Load Baseline
# ============================================================

baseline = SmallCNN().to(device)
baseline.load_state_dict(
    torch.load(
        "mnist_fp32.pth",
        map_location=device,
        weights_only=True,
    )
)
baseline.eval()

ratios = [0.0, 0.3, 0.5, 0.7, 0.9]
results = []


# ============================================================
# 5. Pruning Sweep
# ============================================================

for ratio in ratios:
    model = copy.deepcopy(baseline)
    targets = pruning_targets(model)

    if ratio > 0:
        prune.global_unstructured(
            targets,
            pruning_method=prune.L1Unstructured,
            amount=ratio,
        )

        for module, name in targets:
            prune.remove(module, name)

    actual_sparsity = measure_sparsity(model)
    accuracy = evaluate_accuracy(model)
    latency = measure_latency(model)
    parameter_count, nonzero_parameter_count = count_parameters(model)

    output_path = f"mnist_pruned_{int(ratio * 100):02d}.pth"
    torch.save(model.state_dict(), output_path)
    file_size_kb = os.path.getsize(output_path) / 1024.0

    results.append({
        "Pruning (%)": int(ratio * 100),
        "Actual Sparsity (%)": actual_sparsity,
        "Accuracy (%)": accuracy,
        "Mean Latency (ms)": latency["mean"],
        "Median Latency (ms)": latency["median"],
        "Std Latency (ms)": latency["std"],
        "Parameters": parameter_count,
        "Nonzero Parameters": nonzero_parameter_count,
        "File Size (KB)": file_size_kb,
    })


# ============================================================
# 6. Save and Print Results
# ============================================================

df = pd.DataFrame(results)
df.to_csv("unstructured_pruning_results.csv", index=False)

print()
print("=" * 110)
print("Unstructured Pruning Results")
print("=" * 110)
print(df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print()
print("Saved: unstructured_pruning_results.csv")


# ============================================================
# 7. Visualization
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(
    df["Actual Sparsity (%)"],
    df["Accuracy (%)"],
    marker="o",
)
axes[0].set_xlabel("Actual Weight Sparsity (%)")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Accuracy vs. Sparsity")
axes[0].grid(alpha=0.3)

axes[1].errorbar(
    df["Actual Sparsity (%)"],
    df["Mean Latency (ms)"],
    yerr=df["Std Latency (ms)"],
    marker="o",
    capsize=4,
)
axes[1].set_xlabel("Actual Weight Sparsity (%)")
axes[1].set_ylabel("Single-image Latency (ms)")
axes[1].set_title("Latency vs. Sparsity")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("unstructured_pruning_results.png", dpi=200)

print("Saved: unstructured_pruning_results.png")
PY
```

## 11. Pruning Sweep 실행

```bash
python pruning_sweep.py
```

프로그램은 다음 파일을 생성합니다.

```text
unstructured_pruning_lab/
├─ pruning_env/
├─ data/
├─ check_gpu.py
├─ model.py
├─ train_baseline.py
├─ pruning_sweep.py
├─ mnist_fp32.pth
├─ mnist_pruned_00.pth
├─ mnist_pruned_30.pth
├─ mnist_pruned_50.pth
├─ mnist_pruned_70.pth
├─ mnist_pruned_90.pth
├─ test_data.npy
├─ test_labels.npy
├─ unstructured_pruning_results.csv
└─ unstructured_pruning_results.png
```

결과 표는 다음과 같은 형태로 출력됩니다.

```text
 Pruning (%)  Actual Sparsity (%)  Accuracy (%)  Mean Latency (ms)  ...
           0                 0.00         xx.xx               x.xxx  ...
          30                30.00         xx.xx               x.xxx  ...
          50                50.00         xx.xx               x.xxx  ...
          70                70.00         xx.xx               x.xxx  ...
          90                90.00         xx.xx               x.xxx  ...
```

> [!NOTE]
> 정확도와 latency는 장비 상태, 학습 결과, 전력 모드, 온도 및 백그라운드 작업에 따라 달라집니다. 예시 숫자를 정답처럼 사용하지 말고 각 장비에서 직접 측정한 값을 기록하세요.

## 12. 실행 중 자원 사용량 확인

별도의 Jetson 터미널을 열고 가상환경과 관계없이 다음 명령을 실행합니다.

```bash
tegrastats
```

다음 항목을 관찰합니다.

- CPU 사용률
- `GR3D_FREQ` GPU 사용률
- RAM 사용량
- `VDD_IN` 전력
- CPU 및 GPU 온도

실험 조건을 맞추려면 모든 pruning 비율을 같은 실행에서 연속 측정하고, 다른 프로그램은 가능한 한 종료합니다.

## 13. 결과 기록표

프로그램이 저장한 `unstructured_pruning_results.csv`를 참고하여 다음 표를 작성합니다.

| Pruning | 실제 sparsity | Accuracy | Mean latency | Median latency | Parameters | Nonzero Params | 파일 크기 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% |  |  |  |  |  |  |  |
| 30% |  |  |  |  |  |  |  |
| 50% |  |  |  |  |  |  |  |
| 70% |  |  |  |  |  |  |  |
| 90% |  |  |  |  |  |  |  |

다음 항목도 함께 기록합니다.

```text
Jetson 전력 모드 :
GPU 이름        :
PyTorch 버전    :
CUDA 버전       :
측정 날짜/시간  :
실험 중 VDD_IN  :
실험 중 온도    :
```

## 14. 결과 해석

### 정확도

낮거나 중간 수준의 pruning에서는 정확도가 거의 유지될 수 있습니다. 이는 모델에 중복되거나 중요도가 낮은 파라미터가 존재한다는 뜻입니다.

Pruning 후 정확도가 baseline보다 아주 조금 높게 나올 수도 있습니다. 차이가 매우 작다면 pruning의 확실한 성능 향상으로 단정하지 말고, 학습 및 측정 변동 범위로 해석합니다.

### 파라미터 수와 파일 크기

Unstructured pruning은 weight를 제거하는 대신 기존 dense tensor 안의 값을 0으로 변경합니다. 따라서 tensor shape과 전체 파라미터 수는 그대로입니다.

즉 `Parameters`는 거의 변하지 않지만 `Nonzero Parameters`는 pruning 비율에 따라 감소합니다. 이 차이를 통해 **logical sparsity**와 **실제 tensor 구조 감소**를 구분할 수 있습니다.

일반적인 dense `state_dict`는 0도 다른 실수 값과 같은 방식으로 저장하므로 파일 크기 역시 거의 줄어들지 않을 수 있습니다.

### 실제 latency

일반적인 PyTorch/CUDA dense Conv 및 GEMM kernel은 weight가 0이라는 이유만으로 해당 연산을 자동으로 건너뛰지 않습니다.

따라서 sparsity가 증가해도 latency가 거의 감소하지 않거나 측정 노이즈 범위에서 조금 증가할 수 있습니다.

본 실습의 latency는 각 추론 전후에 `torch.cuda.synchronize()`를 사용하여 측정한 **synchronized end-to-end latency**입니다. 따라서 순수 CUDA kernel 시간만을 의미하지 않으며, Python 호출, framework overhead, kernel launch 및 동기화 비용이 함께 포함됩니다. 모델 간 동일 조건 비교용 지표로 해석합니다.

```text
높은 weight sparsity
        ≠
자동 FLOPs 감소
        ≠
자동 hardware speedup
```

또는 수식으로 다음과 같이 정리할 수 있습니다.

$$
\text{Weight sparsity}
\neq
\text{FLOPs reduction}
\neq
\text{Hardware speedup}
$$

실제 속도 향상을 얻으려면 sparse 연산을 지원하는 전용 kernel 또는 모델의 channel/filter와 tensor shape 자체를 줄이는 structured pruning이 필요합니다.

## 15. 문제 해결

### `RuntimeError: Numpy is not available`

오류 예시:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
RuntimeError: Numpy is not available
```

해결:

```bash
pip uninstall numpy -y
pip install numpy==1.26.4
```

확인:

```bash
python -c "import numpy, torch; print(numpy.__version__); print(torch.tensor([1, 2, 3]).numpy())"
```

### `CUDA available : False`

현재 가상환경에 일반 PC용 PyTorch가 설치되었거나 JetPack/L4T와 wheel 버전이 맞지 않을 수 있습니다. `check_gpu.py`를 다시 실행하고 PyTorch, CUDA 및 L4T 버전을 확인합니다.

```bash
cat /etc/nv_tegra_release
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### `size mismatch` 또는 `Missing key(s)`

`model.py`의 SmallCNN 구조와 `mnist_fp32.pth`를 생성할 때 사용한 모델 구조가 다르다는 뜻입니다. 이 문서의 `model.py`를 변경하지 않았는지 확인하고 baseline을 다시 학습합니다.

### MNIST 파일이 없음

본 배포판은 `torchvision` 자동 다운로드에 의존하지 않습니다. 다음 네 파일이 `data/` 디렉터리에 있는지 확인합니다.

```text
train-images-idx3-ubyte.gz
train-labels-idx1-ubyte.gz
t10k-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz
```

파일이 없다면 교수자가 제공한 MNIST gzip 파일을 `data/`에 복사한 뒤 `python train_baseline.py`를 다시 실행합니다.
