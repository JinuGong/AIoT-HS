# Jetson Structured Pruning 실습

## 1. 이전 실습에서 이어서 진행하기

이 실습은 앞서 완료한 **Jetson Unstructured Pruning 실습의 환경과 파일을 그대로 이어서 사용**합니다. 새로운 가상환경을 만들거나 PyTorch, NumPy 등의 핵심 패키지를 다시 설치하지 않습니다.

먼저 기존 작업 디렉터리로 이동하고 가상환경을 활성화합니다.

```bash
cd ~/unstructured_pruning_lab
source pruning_env/bin/activate
```

다음 파일이 있는지 확인합니다.

```bash
ls -lh model.py mnist_fp32.pth test_data.npy test_labels.npy
```

필수 파일의 역할은 다음과 같습니다.

| 파일 | 역할 |
|---|---|
| `model.py` | Baseline SmallCNN 구조 |
| `mnist_fp32.pth` | 이전 실습에서 직접 학습한 baseline 가중치 |
| `test_data.npy` | 동일한 전처리가 적용된 MNIST 시험 이미지 |
| `test_labels.npy` | MNIST 시험 레이블 |

하나라도 없다면 먼저 [Jetson Unstructured Pruning 실습](./Week5_Jetson_Unstructured_Pruning_실습.md)을 완료한 뒤 돌아옵니다.

> [!IMPORTANT]
> 이 문서의 모든 명령은 `~/unstructured_pruning_lab`에서 `(pruning_env)`가 활성화된 상태로 실행합니다.

## 2. 실습 목표

앞 실습의 unstructured pruning은 개별 weight를 0으로 만들었지만 tensor shape은 바꾸지 않았습니다. 이번에는 convolution channel을 실제로 제거하여 네트워크 구조 자체를 작게 만듭니다.

```text
Unstructured pruning
개별 weight를 0으로 변경
    ↓
Tensor shape 유지
    ↓
일반 dense 연산의 FLOPs는 그대로

Structured pruning
Channel/filter 자체를 제거
    ↓
Tensor shape 축소
    ↓
파라미터와 FLOPs 감소
```

Conv1과 Conv2의 **출력 channel 수를 각각 25%, 50%, 75% 감소**시키는 structured pruning을 적용하고, 각 모델을 3 epoch fine-tuning한 뒤 다음 항목을 비교합니다.

- Convolution channel 수
- 파라미터 수
- FLOPs
- Pruning 직후 정확도
- Fine-tuning 후 정확도
- Jetson single-image latency
- Accuracy–FLOPs Pareto 관계

## 3. Channel 구성

Baseline SmallCNN의 convolution 구조는 다음과 같습니다.

```text
Conv1: 1  → 16
Conv2: 16 → 32
FC1  : 32 × 7 × 7 → 128
FC2  : 128 → 10
```

이 문서에서 `Structured 25%`, `50%`, `75%`는 **각 convolution layer의 출력 channel을 해당 비율만큼 제거한다는 의미**입니다. 전체 파라미터 수나 전체 FLOPs가 같은 비율로 감소한다는 뜻은 아닙니다.

| 설정 | Conv1 출력 | Conv2 출력 | FC1 입력 크기 |
|---|---:|---:|---:|
| Baseline | 16 | 32 | `32 × 7 × 7` |
| Structured 25% | 12 | 24 | `24 × 7 × 7` |
| Structured 50% | 8 | 16 | `16 × 7 × 7` |
| Structured 75% | 4 | 8 | `8 × 7 × 7` |

이번 실습에서는 각 convolution filter의 L1 norm을 channel 중요도로 사용합니다.

$$
I_c=\sum_{i,j,k}|W_{c,i,j,k}|
$$

L1 norm이 큰 출력 channel을 남기고 나머지를 제거합니다. Conv1의 출력 channel을 제거하면 Conv2의 대응하는 입력 channel도 함께 제거해야 합니다. Conv2의 출력 channel을 제거하면 FC1에서 해당 feature map과 연결된 입력 열도 함께 제거해야 합니다.

```text
Conv1 출력 channel 선택
        ↓
Conv2의 대응 입력 channel 선택
        ↓
Conv2 출력 channel 선택
        ↓
FC1의 대응 feature 입력 선택
```

## 4. Structured Pruning Sweep 스크립트 생성

다음 명령으로 `structured_sweep.py`를 생성합니다.

```bash
cat > structured_sweep.py <<'PY'
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
import gzip
import struct
from pathlib import Path

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
    raise RuntimeError("CUDA is not available. Check the previous lab setup.")

print("Device:", device)
print("GPU   :", torch.cuda.get_device_name(0))


# ============================================================
# 2. Required Files
# ============================================================

required_files = [
    "model.py",
    "mnist_fp32.pth",
    "test_data.npy",
    "test_labels.npy",
]

for path in required_files:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file: {path}. "
            "Complete the unstructured pruning lab first."
        )


# ============================================================
# 3. Training and Test Data
# ============================================================

# Use exactly the same preprocessing as the baseline lab:
# uint8 MNIST pixels -> float32 in [0, 1].

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

train_images_path = data_dir / "train-images-idx3-ubyte.gz"
train_labels_path = data_dir / "train-labels-idx1-ubyte.gz"

if not train_images_path.exists():
    raise FileNotFoundError(
        f"Missing MNIST file: {train_images_path}. "
        "Complete the previous lab data setup first."
    )

if not train_labels_path.exists():
    raise FileNotFoundError(
        f"Missing MNIST file: {train_labels_path}. "
        "Complete the previous lab data setup first."
    )

train_images = load_images(train_images_path)
train_labels = load_labels(train_labels_path)

train_dataset = TensorDataset(
    torch.tensor(train_images, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0,
)

# test_data.npy / test_labels.npy were created in the previous lab
# using the same [0,1] preprocessing.
x_test = torch.tensor(
    np.load("test_data.npy"),
    dtype=torch.float32,
)

y_test = torch.tensor(
    np.load("test_labels.npy"),
    dtype=torch.long,
)

print("Train samples:", len(train_dataset))
print("Test X shape :", tuple(x_test.shape))
print("Test Y shape :", tuple(y_test.shape))


# ============================================================
# 4. Structured Model
# ============================================================

class PrunedCNN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# 5. Accuracy
# ============================================================

def evaluate(model):
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


# ============================================================
# 6. Latency
# ============================================================

def measure_latency(model, warmup=100, repeat=500):
    """Measure synchronized end-to-end single-image latency."""
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


# ============================================================
# 7. Parameter Count
# ============================================================

def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


# ============================================================
# 8. FLOPs
# ============================================================

def calculate_flops(model):
    """Count multiply and add as two operations."""

    flops = 0
    hooks = []

    def conv_hook(module, inputs, output):
        nonlocal flops

        batch_size = output.shape[0]
        output_channels = output.shape[1]
        output_height = output.shape[2]
        output_width = output.shape[3]
        kernel_height, kernel_width = module.kernel_size
        input_channels = module.in_channels // module.groups

        flops += (
            batch_size
            * output_channels
            * output_height
            * output_width
            * input_channels
            * kernel_height
            * kernel_width
            * 2
        )

    def linear_hook(module, inputs, output):
        nonlocal flops
        batch_size = output.shape[0] if output.dim() > 1 else 1
        flops += batch_size * module.in_features * module.out_features * 2

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    dummy = torch.randn(1, 1, 28, 28, device=device)

    model.eval()
    with torch.no_grad():
        _ = model(dummy)

    for hook in hooks:
        hook.remove()

    return int(flops)


# ============================================================
# 9. Create a Structured-Pruned Model
# ============================================================

def create_pruned_model(original, c1, c2):
    new_model = PrunedCNN(c1, c2).to(device)

    # Conv1: select output filters by L1 norm.
    conv1 = original.features[0]
    importance1 = conv1.weight.detach().abs().sum(dim=(1, 2, 3))
    keep1 = torch.argsort(importance1, descending=True)[:c1]
    keep1 = torch.sort(keep1).values

    with torch.no_grad():
        new_model.features[0].weight.copy_(conv1.weight[keep1])
        new_model.features[0].bias.copy_(conv1.bias[keep1])

    # Conv2: first retain the Conv1 input channels, then rank outputs.
    conv2 = original.features[3]
    reduced_input = conv2.weight.detach()[:, keep1, :, :]
    importance2 = reduced_input.abs().sum(dim=(1, 2, 3))
    keep2 = torch.argsort(importance2, descending=True)[:c2]
    keep2 = torch.sort(keep2).values

    with torch.no_grad():
        new_model.features[3].weight.copy_(
            conv2.weight[keep2][:, keep1, :, :]
        )
        new_model.features[3].bias.copy_(conv2.bias[keep2])

    # FC1: retain all 7x7 features belonging to each kept Conv2 channel.
    feature_indices = []

    for channel in keep2.tolist():
        start = channel * 7 * 7
        end = start + 7 * 7
        feature_indices.extend(range(start, end))

    feature_indices = torch.tensor(
        feature_indices,
        dtype=torch.long,
        device=device,
    )

    fc1 = original.classifier[1]

    with torch.no_grad():
        new_model.classifier[1].weight.copy_(
            fc1.weight[:, feature_indices]
        )
        new_model.classifier[1].bias.copy_(fc1.bias)

        # FC2 shape does not change.
        new_model.classifier[3].weight.copy_(
            original.classifier[3].weight
        )
        new_model.classifier[3].bias.copy_(
            original.classifier[3].bias
        )

    return new_model, keep1.tolist(), keep2.tolist()


# ============================================================
# 10. Fine-Tuning
# ============================================================

def finetune(model, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = []

    for epoch in range(epochs):
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

            running_loss += loss.item() * images.size(0)

        average_loss = running_loss / len(train_dataset)
        accuracy = evaluate(model)
        history.append((average_loss, accuracy))

        print(
            f"  Epoch {epoch + 1}/{epochs}"
            f" | Loss: {average_loss:.4f}"
            f" | Accuracy: {accuracy:.2f}%"
        )

    return history


# ============================================================
# 11. Pareto Test
# ============================================================

def is_dominated(index, dataframe):
    accuracy_i = dataframe.loc[index, "Accuracy After FT (%)"]
    flops_i = dataframe.loc[index, "FLOPs"]

    for other in dataframe.index:
        if other == index:
            continue

        accuracy_j = dataframe.loc[other, "Accuracy After FT (%)"]
        flops_j = dataframe.loc[other, "FLOPs"]

        no_worse = accuracy_j >= accuracy_i and flops_j <= flops_i
        strictly_better = accuracy_j > accuracy_i or flops_j < flops_i

        if no_worse and strictly_better:
            return True

    return False


# ============================================================
# 12. Load Baseline
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


# ============================================================
# 13. Baseline Measurement
# ============================================================

baseline_accuracy = evaluate(baseline)
baseline_latency = measure_latency(baseline)
baseline_parameters = count_parameters(baseline)
baseline_flops = calculate_flops(baseline)

results = [{
    "Model": "Baseline",
    "C1": 16,
    "C2": 32,
    "Parameters": baseline_parameters,
    "FLOPs": baseline_flops,
    "Accuracy Before FT (%)": baseline_accuracy,
    "Accuracy After FT (%)": baseline_accuracy,
    "Mean Latency (ms)": baseline_latency["mean"],
    "Median Latency (ms)": baseline_latency["median"],
    "Std Latency (ms)": baseline_latency["std"],
}]


# ============================================================
# 14. Structured Pruning Sweep
# ============================================================

settings = [
    ("Structured 25%", 12, 24, 25),
    ("Structured 50%", 8, 16, 50),
    ("Structured 75%", 4, 8, 75),
]

for name, c1, c2, ratio in settings:
    print()
    print("=" * 60)
    print(name)
    print(f"Channels: Conv1 16 -> {c1}, Conv2 32 -> {c2}")
    print("=" * 60)

    model, keep1, keep2 = create_pruned_model(baseline, c1, c2)

    print("Kept Conv1 channels:", keep1)
    print("Kept Conv2 channels:", keep2)

    accuracy_before = evaluate(model)
    print(f"Accuracy before fine-tuning: {accuracy_before:.2f}%")

    finetune(model, epochs=3)

    accuracy_after = evaluate(model)
    latency = measure_latency(model)
    parameters = count_parameters(model)
    flops = calculate_flops(model)

    output_path = f"mnist_structured_{ratio}_ft.pth"
    torch.save(model.state_dict(), output_path)
    print("Saved:", output_path)

    results.append({
        "Model": name,
        "C1": c1,
        "C2": c2,
        "Parameters": parameters,
        "FLOPs": flops,
        "Accuracy Before FT (%)": accuracy_before,
        "Accuracy After FT (%)": accuracy_after,
        "Mean Latency (ms)": latency["mean"],
        "Median Latency (ms)": latency["median"],
        "Std Latency (ms)": latency["std"],
    })


# ============================================================
# 15. Results and Pareto Front
# ============================================================

df = pd.DataFrame(results)
df["Parameter Reduction (%)"] = (
    100.0 * (1.0 - df["Parameters"] / baseline_parameters)
)
df["FLOPs Reduction (%)"] = (
    100.0 * (1.0 - df["FLOPs"] / baseline_flops)
)

df["Pareto"] = True

for index in df.index:
    if is_dominated(index, df):
        df.loc[index, "Pareto"] = False

df.to_csv("structured_sweep_results.csv", index=False)

print()
print("=" * 120)
print("STRUCTURED PRUNING RESULTS")
print("=" * 120)
print(df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print()
print("Saved: structured_sweep_results.csv")


# ============================================================
# 16. Visualization
# ============================================================

pareto_df = df[df["Pareto"]].sort_values("FLOPs")

plt.figure(figsize=(8, 6))

plt.scatter(
    df["FLOPs"] / 1_000_000,
    df["Accuracy After FT (%)"],
    s=80,
    label="Models",
)

for _, row in df.iterrows():
    plt.annotate(
        row["Model"],
        (row["FLOPs"] / 1_000_000, row["Accuracy After FT (%)"]),
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.plot(
    pareto_df["FLOPs"] / 1_000_000,
    pareto_df["Accuracy After FT (%)"],
    marker="o",
    label="Pareto Front",
)

plt.xlabel("FLOPs (million, multiply + add = 2 operations)")
plt.ylabel("Accuracy After Fine-Tuning (%)")
plt.title("Structured Pruning: Accuracy vs. FLOPs")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("structured_accuracy_flops.png", dpi=200)

print("Saved: structured_accuracy_flops.png")
PY
```

생성된 파일을 확인합니다.

```bash
ls -lh structured_sweep.py
```

## 5. 실험 실행

```bash
python structured_sweep.py
```

스크립트는 다음 순서로 동작합니다.

```text
Baseline 측정
    ↓
Structured 25% 모델 생성
    ↓
Pruning 직후 정확도 측정
    ↓
3 epoch fine-tuning
    ↓
Accuracy / Params / FLOPs / Latency 측정
    ↓
Structured 50%에서 반복
    ↓
Structured 75%에서 반복
    ↓
CSV와 Pareto 그래프 저장
```

Fine-tuning과 latency 반복 측정이 포함되므로 완료까지 시간이 걸릴 수 있습니다. 실행 중에는 터미널을 종료하지 않습니다.

## 6. 생성 파일

실행이 완료되면 다음 파일이 생성됩니다.

```text
mnist_structured_25_ft.pth
mnist_structured_50_ft.pth
mnist_structured_75_ft.pth
structured_sweep_results.csv
structured_accuracy_flops.png
```

파일을 확인합니다.

```bash
ls -lh mnist_structured_*_ft.pth structured_sweep_results.csv structured_accuracy_flops.png
```

## 7. 결과 표 확인

터미널에는 다음 형태의 표가 출력됩니다.

```text
Model           C1  C2  Parameters  FLOPs    Accuracy Before FT  Accuracy After FT  Mean Latency  Pareto
Baseline        16  32  ...         ...      ...                 ...                ...           ...
Structured 25%  12  24  ...         ...      ...                 ...                ...           ...
Structured 50%   8  16  ...         ...      ...                 ...                ...           ...
Structured 75%   4   8  ...         ...      ...                 ...                ...           ...
```

저장된 CSV를 터미널에서 확인할 수도 있습니다.

```bash
column -s, -t structured_sweep_results.csv | less -S
```

`less`를 종료하려면 `q`를 누릅니다.

## 8. FLOPs 계산 기준

이 실습에서는 multiplication과 addition을 각각 한 번의 연산으로 계산합니다.

Convolution의 FLOPs는 다음과 같이 계산합니다.

$$
2\times C_{out}\times H_{out}\times W_{out}
\times C_{in}\times K_h\times K_w
$$

Linear 계층의 FLOPs는 다음과 같습니다.

$$
2\times N_{in}\times N_{out}
$$

다른 도구는 multiply-accumulate 한 쌍을 1 MAC으로 계산할 수 있습니다. 따라서 외부 도구와 비교할 때 값이 약 2배 차이 날 수 있으므로 계산 기준을 함께 기록해야 합니다.

또한 본 코드의 FLOPs 계산은 주로 `Conv2d`와 `Linear`의 multiply/add를 대상으로 하며, ReLU, pooling, bias addition 등은 포함하지 않습니다. 따라서 절대적인 하드웨어 연산량이라기보다 **모델 간 동일 기준 비교 지표**로 사용합니다.

## 9. Fine-Tuning의 역할

Structured pruning 직후에는 feature channel 자체가 사라지므로 정확도가 크게 떨어질 수 있습니다.

```text
Baseline
    ↓
Channel 제거
    ↓
표현 능력 및 정확도 감소
    ↓
Fine-tuning
    ↓
남은 파라미터 재조정
    ↓
정확도 회복
```

Fine-tuning은 제거된 channel을 복원하지 않습니다. 작은 네트워크 구조를 유지한 상태에서 남아 있는 weight만 다시 학습합니다. 따라서 파라미터와 FLOPs 감소 효과는 유지됩니다.

## 10. 결과 해석

### Unstructured pruning과의 차이

| 방법 | Weight 0 증가 | Tensor shape 감소 | Dense FLOPs 감소 | 정확도 회복 방법 |
|---|---:|---:|---:|---|
| Unstructured pruning | 예 | 아니요 | 일반적으로 아니요 | 선택적 fine-tuning |
| Structured pruning | 개별 weight 0화가 핵심 아님 | 예 | 예 | Fine-tuning 권장 |

### FLOPs와 latency

Structured pruning은 channel과 tensor shape을 실제로 줄이므로 파라미터와 FLOPs가 감소합니다.

```text
Channel 감소
    ↓
Tensor shape 감소
    ↓
파라미터 감소
    ↓
FLOPs 감소
```

그러나 FLOPs가 감소했다고 실제 latency가 반드시 같은 비율로 감소하는 것은 아닙니다.

MNIST SmallCNN처럼 매우 작은 모델에서는 다음 고정 비용의 비중이 큽니다.

- CUDA kernel launch overhead
- PyTorch framework overhead
- CPU–GPU 명령 전달
- 메모리 접근
- 동기화 비용

따라서 FLOPs가 크게 감소해도 Jetson latency는 거의 같게 측정될 수 있습니다.

본 실습의 latency는 각 추론 전후에 `torch.cuda.synchronize()`를 호출하여 측정한 **synchronized end-to-end latency**입니다. 따라서 순수 CUDA kernel 실행시간만을 의미하지 않으며 Python 호출, framework overhead, kernel launch 및 동기화 비용이 함께 포함됩니다. 모델 간 동일 조건 비교용으로 해석합니다.

$$
\text{FLOPs reduction}
\not\Rightarrow
\text{proportional latency reduction}
$$

이 결과는 structured pruning이 실패했다는 의미가 아닙니다. 모델 구조와 이론적 연산량은 실제로 감소했지만, 현재 모델 규모에서는 고정 overhead가 그 차이를 가릴 수 있다는 뜻입니다.

## 11. Accuracy–FLOPs Pareto 분석

이번 실습의 두 목적은 다음과 같습니다.

$$
\max \operatorname{Accuracy}
$$

$$
\min \operatorname{FLOPs}
$$

어떤 모델 A보다 정확도가 낮거나 같으면서 FLOPs가 더 많거나 같은 모델 B는 A에 의해 지배됩니다.

```text
모델 A가 모델 B보다
Accuracy는 같거나 높고
FLOPs는 같거나 낮으며
둘 중 하나는 확실히 더 좋음
        ↓
모델 B는 dominated
```

지배되지 않는 모델들은 이 실험에서의 **Pareto-optimal candidates**입니다. `structured_accuracy_flops.png`에서 Accuracy–FLOPs trade-off와 Pareto front를 확인합니다.

모델 선택에는 하나의 정답만 있는 것이 아니라 배포 조건이 필요합니다.

- Accuracy 98% 이상을 유지하면서 FLOPs가 가장 작은 모델
- Baseline 대비 정확도 손실을 0.5%p 이하로 유지하는 가장 작은 모델
- 전력이나 메모리가 가장 제한된 환경에서 사용할 모델

## 12. Unstructured 결과와 함께 비교하기

앞 실습에서 생성한 `unstructured_pruning_results.csv`가 있다면 다음 기준으로 결과를 나란히 비교합니다.

| 방법 | 구조 | 파라미터 수 | Nonzero weight | Dense FLOPs | Accuracy | Latency |
|---|---|---:|---:|---:|---:|---:|
| Baseline | 16 → 32 | 측정값 | 측정값 | 측정값 | 측정값 | 측정값 |
| Unstructured 50% | 16 → 32 | 동일 | 약 50% | 동일 | 측정값 | 측정값 |
| Structured 50% + FT | 8 → 16 | 감소 | 구조에 맞게 감소 | 감소 | 측정값 | 측정값 |

최종적으로 다음 세 문장으로 정리할 수 있습니다.

1. Unstructured pruning은 zero weight를 늘리지만 일반 dense 연산의 shape과 FLOPs를 바꾸지 않습니다.
2. Structured pruning은 channel과 tensor shape을 줄여 파라미터와 FLOPs를 실제로 줄입니다.
3. 모델이 매우 작으면 FLOPs 감소가 Jetson latency 감소로 바로 나타나지 않을 수 있습니다.

## 13. 문제 해결

### `Missing required file`

앞 실습에서 생성할 파일이 없거나 다른 디렉터리에서 명령을 실행한 경우입니다.

```bash
pwd
ls -lh model.py mnist_fp32.pth test_data.npy test_labels.npy
```

현재 위치가 `~/unstructured_pruning_lab`인지 확인합니다.

### MNIST 학습 파일이 없음

앞 실습에서 준비한 MNIST gzip 파일이 삭제되었거나 `data/` 경로가 다른 경우입니다.

다음 네 파일을 확인합니다.

```bash
ls -lh \
  data/train-images-idx3-ubyte.gz \
  data/train-labels-idx1-ubyte.gz \
  data/t10k-images-idx3-ubyte.gz \
  data/t10k-labels-idx1-ubyte.gz
```

이 실습은 `torchvision` 자동 다운로드에 의존하지 않습니다. 파일이 없다면 교수자가 제공한 MNIST gzip 파일을 `data/`에 복사한 뒤 `python structured_sweep.py`를 다시 실행합니다.

### CUDA를 사용할 수 없음

```bash
python check_gpu.py
```

앞 실습에서 사용한 가상환경이 활성화되어 있는지 확인합니다.

```bash
which python
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Fine-tuning 후 정확도가 충분히 회복되지 않음

- Baseline accuracy가 정상인지 확인합니다.
- 학습과 평가가 모두 동일한 `[0,1]` 전처리(`x / 255.0`)를 사용하는지 확인합니다.
- `model.py`의 구조를 변경하지 않았는지 확인합니다.
- 3 epoch 결과를 먼저 기록한 뒤, 추가 실험으로 epoch 수를 늘려 비교할 수 있습니다.
- 학습 결과에는 seed와 하드웨어에 따른 변동이 있을 수 있습니다.

### Latency 차이가 거의 없음

작은 MNIST CNN에서는 정상적으로 나타날 수 있는 결과입니다. Mean latency 하나만 보지 말고 median과 standard deviation을 함께 비교합니다. 모든 모델을 동일한 전력 모드와 온도 조건에서 측정했는지도 확인합니다.

## 13.5. 핵심 정리

이번 실습의 핵심은 다음 세 가지입니다.

1. Structured pruning은 channel 수와 tensor shape을 실제로 줄이므로 파라미터와 dense FLOPs를 감소시킬 수 있습니다.
2. Pruning 직후 accuracy는 하락할 수 있지만, fine-tuning을 통해 작은 구조를 유지한 채 상당 부분 회복할 수 있습니다.
3. FLOPs 감소와 Jetson end-to-end latency 감소는 동일한 비율로 나타나지 않을 수 있습니다.

> [!IMPORTANT]
> 이 실습에서 `Structured 50%`는 Conv1/Conv2의 출력 channel을 각각 절반으로 줄였다는 뜻입니다. 전체 FLOPs가 정확히 50% 감소했다는 뜻은 아닙니다.