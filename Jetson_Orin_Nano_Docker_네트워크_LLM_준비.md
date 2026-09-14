# Jetson Orin Nano Docker 네트워크 설정 및 LLM 실행 준비

이 문서는 Docker/NVIDIA 환경을 확인한 이후 인터넷 연결 문제를 진단하고, 유선 네트워크와 DNS를 설정하여 Docker 이미지 다운로드를 정상화한 과정을 정리한 실습 노트입니다. 마지막에는 Jetson용 `llama.cpp` 컨테이너 이미지를 준비하고 LLM 실습으로 이어지는 전체 구조를 설명합니다.

> [!IMPORTANT]
> 실제 IP 주소, 게이트웨이, DNS 서버 및 기타 네트워크 식별 정보는 보안을 위해 모두 자리표시자로 변경했습니다. 명령을 실행하기 전에 반드시 해당 기관이나 실습 환경에서 안내받은 값으로 바꾸세요.

## 네트워크 자리표시자

| 자리표시자 | 의미 |
|---|---|
| `{TEST_IP}` | 인터넷 연결 시험용 공용 IP |
| `{WIFI_IP}` | Jetson에 할당된 Wi-Fi IP |
| `{WIFI_PREFIX}` | Wi-Fi 네트워크의 CIDR prefix |
| `{WIFI_GATEWAY}` | Wi-Fi 기본 게이트웨이 |
| `{PC_WIRED_IP}` | Windows PC의 유선 IP |
| `{JETSON_WIRED_IP}` | Jetson에 배정할 유선 고정 IP |
| `{WIRED_PREFIX}` | 유선 네트워크의 CIDR prefix |
| `{WIRED_NETMASK}` | 유선 네트워크의 서브넷 마스크 |
| `{WIRED_GATEWAY}` | 유선 네트워크의 기본 게이트웨이 |
| `{DNS_PRIMARY}` | 기본 DNS 서버 |
| `{DNS_SECONDARY}` | 보조 DNS 서버 |
| `{JETSON_USB_IP}` | USB-C 네트워크에서 사용하는 Jetson 주소 |
| `{USB_GATEWAY}` | USB-C 네트워크의 게이트웨이 주소 |

## 전체 진행 흐름

```text
Docker/NVIDIA 환경 확인
    ↓
Docker 사용자 권한 설정
    ↓
이미지 다운로드 실패 확인
    ↓
Jetson 호스트의 인터넷 및 DNS 진단
    ↓
유선랜 고정 IP와 DNS 설정
    ↓
Docker 이미지 다운로드 확인
    ↓
Jetson용 llama.cpp 이미지 다운로드
    ↓
LLM 컨테이너 실행 준비
```

## 1. Docker 설치 상태 확인

Jetson에 Docker와 NVIDIA Container Toolkit이 이미 설치되어 있는지 확인합니다.

```bash
docker --version
nvidia-ctk --version
```

실습 장비에서 확인된 버전:

```text
Docker 29.1.3
NVIDIA Container Toolkit 1.16.2
```

NVIDIA Runtime 등록 상태도 확인합니다.

```bash
docker info | grep -i runtime
```

확인된 결과:

```text
Runtimes: io.containerd.runc.v2 nvidia runc
Default Runtime: runc
```

기본 Runtime이 `runc`인 것은 문제가 아닙니다. GPU가 필요한 컨테이너를 실행할 때 NVIDIA Runtime을 명시하면 됩니다.

```bash
docker run --runtime=nvidia [기타 옵션] 이미지이름
```

## 2. Docker를 `sudo` 없이 사용하도록 설정

처음에는 일반 사용자에게 Docker daemon 접근 권한이 없어 다음 오류가 발생했습니다.

```text
permission denied while trying to connect to the docker API
```

현재 사용자를 `docker` 그룹에 추가합니다.

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

권한 적용 후 다음 명령이 `sudo` 없이 실행되는지 확인합니다.

```bash
docker info
```

> [!CAUTION]
> `docker` 그룹 사용자는 호스트에서 사실상 관리자 수준의 작업을 수행할 수 있습니다. 개인 또는 실습용 Jetson에서는 편리하지만, 공용 서버에서는 신뢰할 수 있는 계정에만 권한을 부여해야 합니다.

## 3. Docker 이미지 다운로드 실패

Docker 설치 상태를 확인하기 위해 테스트 컨테이너를 실행합니다.

```bash
docker run --rm hello-world
```

처음에는 다음과 같은 오류가 발생했습니다.

```text
dial tcp: lookup registry-1.docker.io on [LOCAL_DNS_STUB]:53:
i/o timeout
```

이는 Docker Engine 고장보다는 DNS 조회 실패 또는 Jetson 호스트의 인터넷 연결 문제일 가능성이 높다는 의미입니다.

## 4. Jetson 인터넷 연결 확인

### 4.1 IP 기반 외부 통신 확인

DNS를 거치지 않고 공용 테스트 IP로 통신이 가능한지 확인합니다.

```bash
ping -c 3 {TEST_IP}
```

이 테스트는 실패했습니다.

### 4.2 도메인 이름 확인

```bash
ping -c 3 example.com
```

확인된 오류:

```text
Temporary failure in name resolution
```

추가로 HTTPS 연결과 DNS 질의를 확인합니다.

```bash
curl -I https://example.com
resolvectl query example.com
```

확인된 오류:

```text
Could not resolve host: example.com
```

IP 기반 통신과 도메인 조회가 모두 실패했으므로, Docker 내부가 아니라 Jetson 호스트의 네트워크 또는 DNS 문제로 판단했습니다.

## 5. Wi-Fi 연결 상태 확인

활성 연결과 네트워크 장치 상태를 확인합니다.

```bash
nmcli connection show --active
nmcli device status
```

Wi-Fi 연결은 활성화되어 있었습니다.

```text
[WIFI_PROFILE]    wifi    [WIFI_INTERFACE]
```

라우팅 테이블을 확인합니다.

```bash
ip route
```

Wi-Fi에는 다음과 같은 네트워크 정보가 할당되어 있었습니다.

```text
Jetson Wi-Fi IP : {WIFI_IP}
Gateway         : {WIFI_GATEWAY}
Subnet          : /{WIFI_PREFIX}
```

기본 경로도 설정되어 있었습니다.

```text
default via {WIFI_GATEWAY} dev [WIFI_INTERFACE]
```

즉, Wi-Fi 연결과 IP 주소 할당은 완료되었지만 DNS 또는 외부망 접근이 정상적이지 않은 상태였습니다.

## 6. 학교 유선랜으로 전환

학교에서 별도로 배정받은 Jetson용 유선 IP를 사용할 수 있어 유선랜으로 전환했습니다.

Jetson의 네트워크 장치 상태를 확인합니다.

```bash
nmcli device status
```

확인된 유선 인터페이스:

```text
eno1    ethernet    connecting    Wired connection 1
```

따라서 이 실습 장비의 물리적 유선랜 인터페이스는 `eno1`, NetworkManager 연결 프로필은 `Wired connection 1`입니다.

> [!NOTE]
> 다른 장비에서는 인터페이스 및 연결 프로필 이름이 다를 수 있습니다. 반드시 `nmcli device status`와 `nmcli connection show`에서 실제 이름을 확인하세요.

## 7. Windows PC의 유선 네트워크 정보 확인

Windows PowerShell 또는 명령 프롬프트에서 다음 명령을 실행합니다.

```powershell
ipconfig /all
```

확인해야 할 항목은 다음과 같습니다.

```text
IPv4 주소       : {PC_WIRED_IP}
서브넷 마스크   : {WIRED_NETMASK}
기본 게이트웨이 : {WIRED_GATEWAY}
기본 DNS        : {DNS_PRIMARY}
보조 DNS        : {DNS_SECONDARY}
```

학교에서 별도로 안내하거나 배정한 값을 기준으로 Jetson의 유선 네트워크를 다음과 같이 구성했습니다.

```text
Jetson IP : {JETSON_WIRED_IP}
Subnet    : /{WIRED_PREFIX}
Gateway   : {WIRED_GATEWAY}
```

> [!WARNING]
> PC 주소의 마지막 숫자에 임의로 `1`을 더하는 방식은 다른 장치와 IP 충돌을 일으킬 수 있습니다. 반드시 학교 전산 담당자나 네트워크 관리자가 Jetson용으로 배정한 IP를 사용하세요.

## 8. Jetson 유선랜 고정 IP 설정

자리표시자를 실제로 배정받은 값으로 바꿔 실행합니다.

```bash
sudo nmcli connection modify "Wired connection 1" \
  ipv4.method manual \
  ipv4.addresses "{JETSON_WIRED_IP}/{WIRED_PREFIX}" \
  ipv4.gateway "{WIRED_GATEWAY}"
```

연결을 활성화합니다.

```bash
sudo nmcli connection up "Wired connection 1"
```

유선 인터페이스에 IP가 할당되었는지 확인합니다.

```bash
ip -4 addr show eno1
```

정상 출력 형태:

```text
inet {JETSON_WIRED_IP}/{WIRED_PREFIX}
```

## 9. 유선랜 경로의 우선순위 확인

```bash
ip route
```

라우팅 테이블은 다음과 같은 형태였습니다.

```text
default via {WIRED_GATEWAY} dev eno1 proto static metric [WIRED_METRIC]
default via {WIFI_GATEWAY} dev [WIFI_INTERFACE] proto dhcp metric [WIFI_METRIC]
default via {USB_GATEWAY} dev l4tbr0 metric [USB_METRIC]
```

일반적으로 route metric이 작을수록 우선순위가 높습니다. 이 실습에서는 다음 순서로 외부 경로가 선택되었습니다.

```text
Internet
   ↑
eno1 (유선)             가장 낮은 metric → 1순위
   ↑
[WIFI_INTERFACE]        그다음 metric   → 2순위
   ↑
l4tbr0 (USB-C network)  가장 높은 metric → 3순위
```

따라서 Jetson은 외부 인터넷에 접속할 때 유선랜을 우선 사용합니다.

## 10. DNS 서버 설정

유선 IP와 게이트웨이를 설정한 후에도 다음 명령에서 도메인 조회 오류가 발생했습니다.

```bash
curl -I https://example.com
```

```text
Could not resolve host: example.com
```

Windows의 `ipconfig /all`에서 확인한 기관 DNS를 Jetson 유선 연결에도 설정합니다.

```bash
sudo nmcli connection modify "Wired connection 1" \
  ipv4.ignore-auto-dns yes \
  ipv4.dns "{DNS_PRIMARY} {DNS_SECONDARY}"
```

연결을 재시작하여 설정을 적용합니다.

```bash
sudo nmcli connection down "Wired connection 1"
sudo nmcli connection up "Wired connection 1"
```

적용된 DNS 상태를 확인합니다.

```bash
resolvectl status eno1
```

연결 재시작 중에는 SSH 세션이 잠시 끊길 수 있으므로, 가능하면 Jetson의 로컬 터미널이나 USB-C 관리 연결을 함께 확보한 상태에서 진행합니다.

## 11. Docker 최종 동작 확인

네트워크와 DNS 설정 후 다시 테스트 컨테이너를 실행합니다.

```bash
docker run --rm hello-world
```

이번에는 ARM64용 이미지가 정상적으로 다운로드되었습니다.

```text
latest: Pulling from library/hello-world
[LAYER_ID]: Pull complete
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.
```

이로써 다음 항목이 모두 정상임을 확인했습니다.

```text
Jetson
├─ 유선 인터넷       ✓
├─ DNS               ✓
├─ Docker            ✓
├─ ARM64 이미지 pull ✓
└─ NVIDIA Runtime    ✓
```

## 12. Jetson용 `llama.cpp` 이미지 다운로드

NVIDIA에서 Jetson Orin용으로 제공하는 `llama.cpp` Docker 이미지를 다운로드합니다.

```bash
docker pull ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin
```

이 이미지는 `hello-world`보다 크기가 상당히 크므로 다운로드에 시간이 걸리는 것이 정상입니다. 다운로드 중에는 전원과 네트워크 연결을 안정적으로 유지합니다.

다운로드가 끝나면 이미지가 저장되었는지 확인할 수 있습니다.

```bash
docker image ls
```

## 13. Python 환경과 LLM Docker 환경의 관계

```text
Jetson Ubuntu
│
├─ 기존 Python 환경
│  └─ ~/venvs/jetson
│     └─ OpenCV / YOLO / Flask
│
└─ Docker
   └─ NVIDIA llama.cpp Image
      └─ Container
         └─ GGUF LLM
            ↓
         Jetson GPU
```

Conda의 `(base)` 환경이나 기존 Jetson Python 가상환경과 LLM Docker 환경은 별개입니다. 따라서 기존 YOLO 환경의 패키지 구성을 변경하지 않고도 LLM을 실행할 수 있습니다.

## 14. 현재 네트워크 구조

실제 주소는 모두 자리표시자로 표시했습니다.

```text
┌────────────────────┐
│  Windows Desktop   │
└─────────┬──────────┘
          │ USB-C
          │ {USB_NETWORK}
          ▼
┌────────────────────┐
│ Jetson Orin Nano   │
│ {JETSON_USB_IP}    │
│                    │
│ eno1               │
│ {JETSON_WIRED_IP}  │
└─────────┬──────────┘
          │ Ethernet
          ▼
    School Network
          │
          ▼
       Internet
```

각 연결의 역할은 다음과 같습니다.

```text
Windows → Jetson 관리 및 개발
    USB-C → SSH / VS Code 등의 로컬 연결

Jetson → 외부 서비스
    eno1 → 학교망 → Docker Registry / GitHub / 모델 저장소 등
```

Wi-Fi가 원천적으로 동작하지 않는 것은 아닙니다. Wi-Fi에서 IP 주소를 받은 상태였으므로, 기관에서 허용하는 DNS와 인증 정책을 확인한 뒤 다시 시험할 수 있습니다. 다만 LLM 실습 중에는 이미 정상 동작이 확인된 유선 환경을 유지하는 편이 안정적입니다.

## 15. 다음 단계

`llama.cpp` 이미지 다운로드가 끝나면 다음 순서로 실습을 진행합니다.

```text
① llama.cpp 컨테이너 실행
        ↓
② 경량 GGUF 모델 다운로드
        ↓
③ Jetson GPU에서 LLM 추론
        ↓
④ llama-server 실행
        ↓
⑤ Windows 브라우저에서 접속
        ↓
⑥ tegrastats로 GPU·RAM·전력 사용량 관찰
```

앞선 YOLO 실습과 이번 LLM 실습의 처리 구조를 비교하면 다음과 같습니다.

| 실습 | 주요 처리 흐름 |
|---|---|
| 영상 객체 탐지 | Camera → TensorRT → Jetson GPU inference |
| 온디바이스 LLM | LLM → GGUF/Quantization → llama.cpp → Jetson GPU inference |

두 실습을 통해 Jetson 기반 Edge AI에서 영상 모델과 언어 모델이 각각 어떻게 최적화되고 실행되는지 비교할 수 있습니다.

## 빠른 점검표

- [x] Docker와 NVIDIA Container Toolkit 설치 상태 확인
- [x] 일반 사용자의 Docker 접근 권한 설정
- [x] Docker 이미지 다운로드 오류 확인
- [x] Jetson 호스트의 인터넷 및 DNS 문제 진단
- [x] 유선 네트워크 인터페이스 확인
- [x] 배정받은 유선 고정 IP 설정
- [x] 유선 경로의 우선순위 확인
- [x] 기관 DNS 서버 설정
- [x] `hello-world` ARM64 이미지 실행 확인
- [ ] Jetson용 `llama.cpp` 이미지 다운로드 완료 확인
- [ ] NVIDIA Runtime으로 LLM 컨테이너 실행
- [ ] 경량 GGUF 모델 다운로드
- [ ] Jetson GPU 추론 확인
- [ ] `llama-server` 실행 및 브라우저 접속
- [ ] `tegrastats`로 자원 사용량 관찰
