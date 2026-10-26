# Jetson Orin Nano 2대와 Windows PC를 이용한 Flower 연합학습 실습

> 실습 검증일: 2026-10-19  
> 검증 상태: **2-client CIFAR-10 FedAvg, 3 rounds 정상 완료**  
> 구성: Windows PC (Flower SuperLink/ServerApp) + Jetson Orin Nano × 2 (SuperNode/ClientApp)

## 1. 실습 목표

- USB 네트워크를 통해 하나의 Windows PC와 두 Jetson을 각각 연결한다.
- PC에 Flower SuperLink, Jetson 두 대에 SuperNode를 구성한다.
- Jetson CUDA GPU를 이용하여 CIFAR-10 분류 모델을 로컬 학습한다.
- PC에서 FedAvg로 모델을 집계하고 라운드별 학습 손실 및 평가 정확도를 확인한다.

**주의:** 이 문서는 *실제로 성공한 CIFAR-10 기본 Flower PyTorch 템플릿 실험*을 정리한 것이다. 초기 기획이었던 MNIST 및 Non-IID 실험은 아직 수행하지 않았다.

## 2. 장치 및 네트워크 구성

```text
                           Windows PC
                  Flower SuperLink / ServerApp
                       0.0.0.0:9092 (Fleet)
                       0.0.0.0:9093 (Control)
                       0.0.0.0:9091 (ServerAppIo)
                        /               \
                 USB 네트워크         USB 네트워크
                    /                     \
          Jetson 1                        Jetson 2
          192.168.55.1                    192.168.56.1
          Flower SuperNode                Flower SuperNode
          partition-id=0                  partition-id=1
```

| 구분 | Windows PC | Jetson 1 | Jetson 2 |
|---|---|---|---|
| 역할 | FL Server | FL Client | FL Client |
| Jetson 측 USB IP | — | `192.168.55.1` | `192.168.56.1` |
| PC의 USB IP (실습에서 사용) | `192.168.55.100`, `192.168.56.100` | 접속 대상: `192.168.55.100:9092` | 접속 대상: `192.168.56.100:9092` |
| OS | Windows | Ubuntu 22.04, Jetson Linux R36.4.4 | Ubuntu 22.04, Jetson Linux R36.4.7 |
| Python | 3.11 (가상환경) | 3.10.12 | 3.10.12 |
| Flower | 1.22.0 | 1.22.0 | 1.22.0 |
| PyTorch | 2.8.0+cpu | 2.8.0 / CUDA 12.6 | 2.8.0 / CUDA 12.6 |
| torchvision | 0.23.0+cpu | 0.23.0 | 0.23.0 |
| GPU | 집계에는 사용하지 않음 | Orin, CUDA 동작 확인 | Orin, CUDA 동작 확인 |

> `192.168.55.100`, `192.168.56.100`은 이번 실습에서 성공적으로 사용한 PC 측 USB 인터페이스 주소다. 다른 PC에서는 `ipconfig`로 확인한 뒤 수정한다. 두 Jetson은 서로 직접 통신하지 않아도 된다.

### 2.1 USB 연결 확인

Windows PowerShell:

```powershell
ipconfig
ping 192.168.55.1
ping 192.168.56.1
ssh hansungai@192.168.55.1
ssh hansungai@192.168.56.1
```

Windows에서 임시 HTTP 서버를 열어 Jetson → PC 방향 TCP 연결을 검증한다.

```powershell
python -m http.server 5000 --bind 0.0.0.0
```

Jetson 1:

```bash
curl -I --connect-timeout 5 http://192.168.55.100:5000
```

Jetson 2:

```bash
curl -I --connect-timeout 5 http://192.168.56.100:5000
```

**실제 성공 로그:**

```text
192.168.55.1 - - [19/Oct/2026 18:11:09] "HEAD / HTTP/1.1" 200 -
192.168.56.1 - - [19/Oct/2026 18:11:12] "HEAD / HTTP/1.1" 200 -
```

검증 후 Windows에서 `Ctrl+C`로 HTTP 서버를 종료한다. Flower는 별도의 포트(9091~9093)를 사용한다.

### 2.2 동일 Machine ID 및 SSH 호스트 키 주의

두 Jetson은 `machine-id`와 SSH 호스트 키가 동일했지만, Wi-Fi 주소와 네트워크 인터페이스 정보는 달랐다. 이는 이미지 복제 등의 가능성이 있다. 장치 식별은 IP만이 아니라 장치 고유 하드웨어 정보로 별도 검증할 수 있다. 보안 및 관리 관점에서는 장치별 고유 `machine-id`/SSH 호스트 키를 사용하는 것이 바람직하다. **현재 FL 실험 자체는 완료되었다.**

## 3. 소프트웨어 준비

### 3.1 Windows PC: Flower 서버 환경

```powershell
mkdir C:\fl_lab
cd C:\fl_lab
python -m venv fl_env
.\fl_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "flwr==1.22.0"
```

프로젝트 생성:

```powershell
flwr new jetson-fl --framework PyTorch --username hansungai
cd C:\fl_lab\jetson-fl
```

PC의 PyTorch/torchvision은 CPU 빌드로 설치하였다.

```powershell
python -m pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cpu
```

기본 템플릿이 사용하는 데이터셋 패키지 및 확인된 의존성 호환 버전:

```powershell
python -m pip install "flwr-datasets[vision]==0.5.0"
python -m pip install "click==8.1.8" "huggingface-hub==0.34.4"
python -m pip check
flwr --help
```

실제로 `No broken requirements found.`를 확인했다.

### 3.2 Jetson × 2: Flower 및 CUDA PyTorch

**아래 명령은 두 Jetson 각각에서 실행한다.**

```bash
mkdir -p ~/fl_lab
cd ~/fl_lab
python3 -m venv fl_env
source fl_env/bin/activate
python -m pip install --upgrade pip
python -m pip install "flwr==1.22.0"
python -m pip install "numpy==1.26.4"
python -m pip install "torch==2.8.0" --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
python -m pip install "torchvision==0.23.0" --no-deps --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

Flower 기본 PyTorch 템플릿의 `task.py`는 `flwr_datasets`와 `torchvision.transforms`를 import한다. Jetson에서도 해당 의존성이 필요한 경우 설치한다.

```bash
python -m pip install "flwr-datasets[vision]==0.5.0"
```

**환경 보존 주의:** Jetson에서 `torchvision` 설치 시 `--no-deps`를 사용하여 정상 작동하는 CUDA 지원 `torch`가 다른 빌드로 교체되지 않게 한다. 설치 출처와 빌드 호환성은 운영 환경에 맞게 검증한다.

Jetson Linux/CUDA 확인:

```bash
cat /etc/nv_tegra_release
nvcc --version
```

실제 장비에서는 Jetson Linux R36.4.4 및 R36.4.7, CUDA 12.6(V12.6.68)이 확인되었다.

GPU 테스트:

```bash
python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device='cuda')
y = torch.randn(1024, 1024, device='cuda')
z = x @ y
torch.cuda.synchronize()
print('Matrix multiplication:', z.shape)
print('Result device:', z.device)
print('GPU TEST SUCCESS')
PY
```

**두 Jetson 실제 확인 결과:**

```text
PyTorch: 2.8.0
CUDA version: 12.6
CUDA available: True
GPU: Orin
Matrix multiplication: torch.Size([1024, 1024])
Result device: cuda:0
GPU TEST SUCCESS
```

torchvision 검사:

```bash
python - <<'PY'
import torch, torchvision
from torchvision.transforms import Compose, Normalize, ToTensor
print('PyTorch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
print('SUCCESS')
PY
```

실제 두 Jetson에서 정상 import와 CUDA 활성화를 확인했다.

## 4. Flower 프로젝트 구성

`flwr new`로 생성된 PyTorch 예제 구조:

```text
C:\fl_lab\jetson-fl\
├── pyproject.toml
└── jetson_fl\
    ├── __init__.py
    ├── client_app.py
    ├── server_app.py
    └── task.py
```

- `server_app.py`: FedAvg 전략 설정, 라운드 관리, 글로벌 모델 저장
- `client_app.py`: 각 Jetson에서 학습/평가 실행
- `task.py`: 모델, CIFAR-10 데이터 처리 및 학습/평가 함수

### 4.1 사용된 설정

실제 관찰된 실행 설정:

```toml
[tool.flwr.app.config]
num-server-rounds = 3
fraction-train = 0.5
local-epochs = 1
lr = 0.01

[tool.flwr.federations.jetson-deployment]
address = "127.0.0.1:9093"
insecure = true
```

템플릿의 `[project].dependencies`에는 원래 `torch==2.7.1`, `torchvision==0.22.1`이 들어 있었으나, Jetson CUDA PyTorch 2.8.0과 충돌할 수 있으므로 다음과 같이 **장치별 바이너리를 별도로 관리하는 방식**을 권장한다.

```toml
[project]
name = "jetson-fl"
version = "1.0.0"
description = ""
license = "Apache-2.0"
dependencies = [
  "flwr==1.22.0",
  "flwr-datasets[vision]==0.5.0",
]
```

**기록 구분:** 위 의존성 수정은 재현 및 환경 유지에 대한 권장 설정이다. 실제 성공 당시 `pyproject.toml`의 최종 전체 내용은 별도로 보존되지 않았다. 재현 시 `pip check`, `import torch, torchvision, flwr_datasets`를 각 노드에서 검증한다.

## 5. 실행 순서

각 서비스는 별도의 터미널에서 실행하고 **해당 창을 유지한다**.

### 5.1 Windows PC: SuperLink

```powershell
cd C:\fl_lab
.\fl_env\Scripts\Activate.ps1
flower-superlink --insecure
```

예상 로그:

```text
Starting Flower SuperLink
Starting Control API on 0.0.0.0:9093
Starting ServerAppIo API on 0.0.0.0:9091
Starting Fleet API (gRPC-rere) on 0.0.0.0:9092
```

### 5.2 Jetson 1: SuperNode

```bash
cd ~/fl_lab
source fl_env/bin/activate
flower-supernode \
  --insecure \
  --superlink 192.168.55.100:9092 \
  --node-config "partition-id=0 num-partitions=2"
```

### 5.3 Jetson 2: SuperNode

```bash
cd ~/fl_lab
source fl_env/bin/activate
flower-supernode \
  --insecure \
  --superlink 192.168.56.100:9092 \
  --node-config "partition-id=1 num-partitions=2"
```

> 장치별 IP와 `partition-id`를 혼동하지 않는다. 실제 실행 당시 PC에서 두 노드 모두 샘플링되었다.

### 5.4 Windows PC: 학습 작업 제출

SuperLink를 실행 중인 창이 아닌 **새 PowerShell 창**에서:

```powershell
cd C:\fl_lab\jetson-fl
C:\fl_lab\fl_env\Scripts\Activate.ps1
flwr run . jetson-deployment --stream
```

`--stream` 옵션으로 서버 측 라운드별 로그를 확인한다.

### 5.5 종료 순서

`flwr run` 완료 후 결과를 확인한다. 다음 실험을 하지 않을 경우 두 Jetson SuperNode와 PC SuperLink를 각각 `Ctrl+C`로 종료한다. Jetson 기기 자체를 종료하거나 USB 케이블을 분리할 필요는 없다.

## 6. 실제 FedAvg 실험 결과

실험 데이터셋은 **CIFAR-10**이다. 다운로드 로그에서 학습 50,000개, 테스트 10,000개를 확인하였다. 모델은 기본 Flower PyTorch 템플릿에서 초기화된 약 0.24 MB 파라미터 모델이다.

서버 로그에서 확인된 조건:

```text
Number of rounds: 3
ArrayRecord (0.24 MB)
ConfigRecord (train): {'lr': 0.01}
Fraction: train (0.50) | evaluate (1.00)
Minimum nodes: train (2) | evaluate (2)
Minimum available nodes: 2
Weighted by: 'num-examples'
```

**라운드별 집계 실측:**

| Round | Aggregated train loss | Aggregated eval loss | Aggregated eval accuracy | 학습/평가 결과 |
|---:|---:|---:|---:|---|
| 1 | 1.973645 | 2.167954 | 18.08% | 모두 2 results, 0 failures |
| 2 | 1.831110 | 1.806972 | 33.91% | 모두 2 results, 0 failures |
| 3 | 1.743392 | 1.668302 | 37.55% | 모두 2 results, 0 failures |

최종 로그:

```text
Strategy execution finished in 207.44s
Final results:
  Global Arrays: ArrayRecord (0.238 MB)
  Aggregated ClientApp-side Train Metrics:
    1: train_loss=1.9736
    2: train_loss=1.8311
    3: train_loss=1.7434
  Aggregated ClientApp-side Evaluate Metrics:
    1: eval_acc=0.1808, eval_loss=2.1680
    2: eval_acc=0.3391, eval_loss=1.8070
    3: eval_acc=0.3755, eval_loss=1.6683
Saving final model to disk...
```

### 6.1 결과 해석

- 정확도: **18.08% → 33.91% → 37.55%**, 총 **+19.47%p** 상승.
- 평가 손실: **2.1680 → 1.6683** 감소.
- 학습 손실: **1.9736 → 1.7434** 감소.
- 모든 라운드에서 두 클라이언트의 학습·평가 결과 수신에 성공했으며 실패는 0건.
- 전략 실행 시간: **207.44초 (약 3분 27초)**. 이는 전체 전략 수행 시간이므로 GPU 계산 시간 또는 네트워크 통신 시간 단독 측정값이 아니다.
- 정확도 37.55%는 **3라운드의 제한된 초기 학습 결과**이며 수렴 정확도나 최적 성능을 의미하지 않는다.
- `ServerApp-side Evaluate Metrics: {}`는 서버 측 별도 평가 지표가 없었다는 뜻이며 클라이언트 측 집계 평가가 실패했다는 뜻이 아니다.

## 7. 실제 발생한 오류 및 해결

| 증상/오류 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: torch` (PC) | ServerApp 기본 템플릿의 `server_app.py`가 torch import | Windows CPU PyTorch 설치 |
| `ModuleNotFoundError: flwr_datasets` (PC) | `task.py`에서 데이터셋 패키지 import | `flwr-datasets[vision]==0.5.0` 설치 |
| `Secondary flag is not valid for non-boolean flag` | Flower 1.22.0과 Click 8.5.0 호환성 문제 | `click==8.1.8`로 변경 |
| `huggingface-hub 2.2.0 requires click>=8.4.2` | Flower 요구 조건인 `click<8.2.0`과 충돌 | `huggingface-hub==0.34.4`로 조정, `pip check` 검증 |
| `ModuleNotFoundError: torchvision` (PC) | 기본 템플릿 `task.py`에서 torchvision import | Windows CPU torchvision 0.23.0 설치 |
| `Waiting for nodes to connect: 0 connected` | SuperNode 미연결/미실행 | Jetson별 SuperNode 재시작, USB PC 주소와 포트 9092 확인 |
| `aggregate_train: Received 0 results and 2 failures` | Jetson ClientApp import 실패 | Jetson 측 SuperNode traceback 확인 |
| `ModuleNotFoundError: torchvision` (Jetson) | Jetson `fl_env`에 torchvision 없음 | Jetson CUDA 호환 torchvision 0.23.0 설치 (`--no-deps`) |
| `UnboundLocalError: client_app referenced before assignment` | 선행 ClientApp import 실패 후 예외 처리 과정에서 발생한 2차 오류 | 원래 `ModuleNotFoundError` 해결 |
| `yolo_env`에서 torch `+cu130`, CUDA 사용 불가 | 기존 환경 PyTorch CUDA 13.0 빌드와 장치 CUDA 12.6 지원 수준 불일치 | `fl_env`에 CUDA 12.6용 PyTorch 2.8.0 사용 |
| HF Hub unauthenticated warning | 인증 없이 공개 데이터 다운로드 | 데이터 다운로드 성공했으므로 실습 차단 오류 아님; 필요시 HF 토큰 활용 |

### 7.1 점검 명령 모음

Windows PC:

```powershell
python -m pip check
flwr --help
python -c "import torch, torchvision, flwr_datasets; print(torch.__version__, torchvision.__version__)"
```

Jetson:

```bash
source ~/fl_lab/fl_env/bin/activate
python -m pip check
python - <<'PY'
import torch, torchvision
import flwr_datasets
print(torch.__version__, torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
PY
```

## 8. 실습 운영 시 유의사항

1. **USB 서브넷 분리:** Jetson 1은 `55.x`, Jetson 2는 `56.x` 서브넷이지만, 두 노드 모두 PC에 개별적으로 접속하므로 동일한 Wi-Fi에 있을 필요가 없다.
2. **방화벽:** USB에서 PC TCP 9092로 연결이 불가능하면 Windows 방화벽 인바운드 규칙을 확인한다.
3. **패키지 버전:** CPU용 Windows PyTorch와 GPU용 Jetson PyTorch는 설치 인덱스가 다르다. Jetson에 일반 `pip install torch`를 무심코 실행하지 않는다.
4. **보안:** `--insecure`는 TLS를 사용하지 않는다. 통제된 실습 환경에 한정하며 공개 네트워크에서 그대로 사용하지 않는다.
5. **실험 재현성:** 버전, 데이터 분할, 랜덤 시드, 라운드, 로컬 에폭, 모델 가중치, 로그를 보존한다.
6. **캐시와 다운로드:** 처음 실행에서는 Hugging Face CIFAR-10 데이터 다운로드가 필요하다. 뒤의 실험 시간 비교에서는 최초 다운로드 시간을 분리해야 한다.
7. **서버 모델 저장:** `Saving final model to disk...` 로그는 확인했으나 실제 저장 경로 및 파일의 생성 여부는 본 기록에 포함되지 않았으므로 별도 확인한다.

## 9. 후속 실험 계획

### 실험 A: FedAvg 기준선 강화

- `num-server-rounds`: 3 → 10 또는 20
- `local-epochs`: 1과 3 비교
- 라운드별 정확도·손실 시각화
- 클라이언트별 로컬 학습 시간/통신량/서버 집계 시간 구분 측정
- 반복 실험으로 평균 및 분산 산출

### 실험 B: Non-IID 데이터

- CIFAR-10 또는 MNIST 데이터를 IID 및 Dirichlet label-skew (`alpha=0.1, 0.5, 10`)로 분할
- 라벨 분포 히스토그램 기록
- FedAvg 수렴 속도 및 클라이언트 간 성능 격차 비교

### 실험 C: Jetson 성능 분석

- GPU 사용률, 메모리, 전력 (`tegrastats`)
- 로컬 학습 시간 대비 통신 오버헤드
- 모델 크기/라운드 수/로컬 에폭에 따른 성능 및 비용 비교

### 실험 D: 프라이버시 및 공격 기초

- DLG 등 gradient inversion을 별도 통제된 실습 환경에서 수행
- **FedAvg에서 일반적으로 전송되는 모델 업데이트와 DLG에서 가정하는 개별 gradient 접근 권한의 차이**를 명확히 구분
- 샘플 수, 배치 크기, 공개되는 정보 수준에 따른 복원 가능성 비교

## 10. 실습 완료 체크리스트

- [x] Jetson 2대의 USB IP 확인
- [x] PC ↔ Jetson 양방향 접속을 위한 USB 네트워크 확인
- [x] Jetson → PC HTTP 200 OK 확인
- [x] PC Flower SuperLink 실행
- [x] Jetson 2개 SuperNode 연결
- [x] 두 Jetson에서 CUDA PyTorch 행렬곱 성공
- [x] ClientApp 의존성 오류 해결
- [x] FedAvg 총 3라운드 완료
- [x] 매 라운드 train/evaluate 2 results, 0 failures
- [x] CIFAR-10 평가 정확도 37.55% 확인
- [ ] 모델 파일 실제 저장 경로 확인
- [ ] 라운드별 시간·통신량 분리 계측
- [ ] IID/Non-IID 데이터 분할 확장

---

**핵심 성과:** USB 연결된 Jetson Orin Nano 2대에서 실제 ClientApp 학습·평가를 수행하고 Windows PC의 Flower ServerApp에서 FedAvg로 집계했다. 3라운드 후 CIFAR-10 집계 평가 정확도는 **37.55%**, 전체 전략 실행 시간은 **207.44초**였다.
