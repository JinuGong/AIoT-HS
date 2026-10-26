# Jetson Pruning 실습 환경 설정 — 처음부터 다시 구성

본 문서는 Jetson Orin Nano에서 Unstructured / Structured Pruning 실습을 위한
Python 가상환경을 **처음부터 새로 구성**하는 절차입니다.

기준 환경:

```text
Device   : Jetson Orin Nano
OS       : Ubuntu 22.04
CUDA     : 12.6
Python   : 3.10
PyTorch  : 2.5.0a0+872d972e41.nv24.08
GPU CC   : 8.7 (sm_87)
```

> [!IMPORTANT]
> CUDA와 NVIDIA driver는 Jetson 시스템에 이미 설치된 것을 사용합니다.
> 이 실습에서는 CUDA를 새로 설치하지 않습니다.

---

## 1. 기존 가상환경 삭제

기존 `pruning_env`를 사용 중이라면 먼저 빠져나옵니다.

```bash
deactivate 2>/dev/null || true
```

기존 가상환경만 삭제합니다.

```bash
rm -rf ~/unstructured_pruning_lab/pruning_env
```

> `~/unstructured_pruning_lab` 자체는 삭제하지 않습니다.
> 기존 코드, 데이터, 모델 파일은 그대로 보존합니다.

---

## 2. 시스템 환경 확인

```bash
cat /etc/nv_tegra_release
nvcc --version
python3.10 --version
```

CUDA가 다음과 같이 확인되면 정상입니다.

```text
CUDA 12.6
Python 3.10
```

---

## 3. 기본 시스템 패키지 설치

PyTorch 실행에 필요한 기본 패키지를 설치합니다.

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3.10-venv libopenblas-dev
```

---

## 4. cuSPARSELt 설치

PyTorch 24.06 이후 Jetson build는 cuSPARSELt가 필요할 수 있으므로 먼저 설치합니다.

```bash
wget https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb

sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb

sudo cp \
  /var/cusparselt-local-tegra-repo-ubuntu2204-0.8.1/cusparselt-local-tegra-C4CC87E1-keyring.gpg \
  /usr/share/keyrings/

sudo apt-get update
sudo apt-get install -y cusparselt-cuda-12
```

이미 설치되어 있다면 재설치되어도 문제는 없지만,
수업 장비에서는 최초 1회만 설치하면 됩니다.

---

## 5. 새 Python 가상환경 생성

```bash
mkdir -p ~/unstructured_pruning_lab
cd ~/unstructured_pruning_lab

python3.10 -m venv pruning_env
source pruning_env/bin/activate
```

정상적으로 활성화되면 터미널 앞에 다음이 표시됩니다.

```text
(pruning_env)
```

pip을 업데이트합니다.

```bash
python -m pip install --upgrade pip
```

---

## 6. NumPy 버전 고정

Jetson용 PyTorch wheel과의 호환성을 위해 NumPy를 1.26.4로 고정합니다.

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

## 7. Jetson용 PyTorch 설치

일반적인 다음 명령은 사용하지 않습니다.

```bash
pip install torch
```

본 실습에서는 NVIDIA가 제공하는 Jetson용 PyTorch wheel을 직접 설치합니다.

```bash
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

설치:

```bash
pip install --no-cache-dir "$TORCH_INSTALL"
```

> [!NOTE]
> 이 wheel은 Python 3.10 (`cp310`), aarch64 Jetson용입니다.
> 본 실습에서는 `torchvision`을 사용하지 않으므로 설치하지 않습니다.

---

## 8. 실습 패키지 설치

```bash
pip install pandas matplotlib
```

이번 pruning 실습에서는 다음 패키지는 필요하지 않습니다.

```text
torchvision
transformers
```

---

## 9. PyTorch / CUDA 확인

```bash
python - <<'PY'
import numpy
import torch

print("NumPy          :", numpy.__version__)
print("PyTorch        :", torch.__version__)
print("PyTorch CUDA   :", torch.version.cuda)
print("CUDA available :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU            :", torch.cuda.get_device_name(0))
    print("Arch list      :", torch.cuda.get_arch_list())
PY
```

정상 환경에서는 대략 다음과 같이 출력됩니다.

```text
NumPy          : 1.26.4
PyTorch        : 2.5.0a0+872d972e41.nv24.08
PyTorch CUDA   : 12.6
CUDA available : True
GPU            : Orin
```

`Arch list`에 다음 중 하나가 포함되어 있는지도 확인합니다.

```text
sm_87
compute_87
```

---

## 10. 실제 GPU 연산 확인

단순히 CUDA가 보이는 것뿐 아니라 실제 연산까지 확인합니다.

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is not available"

x = torch.randn(100, 100, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("GPU           :", torch.cuda.get_device_name(0))
print("Result device :", y.device)
print("CUDA test     : SUCCESS")
PY
```

정상 출력 예:

```text
GPU           : Orin
Result device : cuda:0
CUDA test     : SUCCESS
```

여기까지 성공하면 pruning 실습 환경 설정은 완료입니다.

---

## 11. 이후 다시 접속할 때

가상환경을 매번 새로 만들 필요는 없습니다.

새 터미널에서는 다음 두 줄만 실행합니다.

```bash
cd ~/unstructured_pruning_lab
source pruning_env/bin/activate
```

---

## 12. Unstructured / Structured Pruning 공통 환경

두 실습 모두 동일한 환경을 사용합니다.

```text
pruning_env
├─ NumPy 1.26.4
├─ PyTorch 2.5.0a0+872d972e41.nv24.08
├─ CUDA 12.6 (Jetson 시스템)
├─ pandas
└─ matplotlib
```

Structured Pruning 실습에서도 새 가상환경을 만들지 않습니다.

```bash
cd ~/unstructured_pruning_lab
source pruning_env/bin/activate
```

---

## 13. 하지 말아야 할 것

환경 문제가 생겼다고 다음을 임의로 실행하지 않습니다.

```bash
pip install torch
pip install torchvision
pip install -U numpy
```

또한 CUDA Toolkit이나 NVIDIA driver를 다시 설치하지 않습니다.

문제가 발생하면 먼저 다음을 확인합니다.

```bash
which python

python -c "import numpy; print(numpy.__version__)"

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

---

## 핵심 요약

```text
Jetson CUDA 12.6        → 기존 시스템 환경 사용
cuSPARSELt              → 시스템에 최초 1회 설치
pruning_env             → Python 3.10으로 새로 생성
NumPy                   → 1.26.4
PyTorch                 → NVIDIA Jetson용 24.08 wheel
torchvision             → 설치하지 않음
pandas / matplotlib     → pip 설치

Unstructured Pruning    → pruning_env
Structured Pruning      → 동일 pruning_env
```
