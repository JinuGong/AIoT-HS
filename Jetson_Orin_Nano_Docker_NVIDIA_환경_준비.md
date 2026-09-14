# Jetson Orin Nano Docker 및 NVIDIA 환경 준비

이 단계에서는 Docker와 NVIDIA Container Toolkit을 새로 설치하지 않습니다. Jetson에 이미 구성되어 있던 Docker/NVIDIA 컨테이너 환경을 확인하고, 일반 사용자가 `sudo` 없이 Docker를 사용할 수 있도록 권한을 설정합니다.

## 1. 확인된 시스템 구성

현재 Jetson의 구성은 다음과 같습니다.

```text
Jetson Orin Nano
├─ L4T R36.4.7
├─ CUDA 12.6
├─ TensorRT 10.3
│
├─ 기존 YOLO Python 환경
│  └─ ~/venvs/jetson
│
└─ 컨테이너 환경
   ├─ Docker 29.1.3
   ├─ NVIDIA Container Toolkit 1.16.2
   └─ NVIDIA Runtime 사용 가능
```

> [!NOTE]
> 위 버전은 현재 실습 장비에서 확인한 값입니다. 다른 Jetson에서는 JetPack/L4T 이미지와 설치 시점에 따라 버전이 달라질 수 있습니다.

## 2. Docker 설치 상태 확인

Docker 버전을 확인합니다.

```bash
docker --version
```

확인된 결과:

```text
Docker version 29.1.3
```

이 결과가 출력되므로 Docker Engine은 이미 설치된 상태입니다.

## 3. NVIDIA Container Toolkit 확인

NVIDIA Container Toolkit CLI 버전을 확인합니다.

```bash
nvidia-ctk --version
```

확인된 결과:

```text
NVIDIA Container Toolkit CLI version 1.16.2
```

NVIDIA Container Toolkit은 컨테이너가 Jetson의 NVIDIA GPU를 사용할 수 있도록 Docker와 NVIDIA Runtime을 연결하는 구성 요소입니다.

## 4. Docker 접근 권한 오류

처음에는 일반 사용자 계정으로 Docker daemon에 접근할 권한이 없어 다음 오류가 발생했습니다.

```text
permission denied while trying to connect to the docker API
```

이 오류는 Docker 또는 NVIDIA Runtime이 설치되지 않았다는 뜻이 아니라, 현재 사용자에게 Docker 소켓 접근 권한이 없다는 뜻입니다.

## 5. NVIDIA Runtime 등록 상태 확인

관리자 권한으로 Docker 정보를 조회합니다.

```bash
sudo docker info | grep -i runtime
```

확인된 결과:

```text
Runtimes: io.containerd.runc.v2 nvidia runc
Default Runtime: runc
```

`Runtimes` 목록에 `nvidia`가 있으므로 NVIDIA Container Runtime이 Docker에 정상적으로 등록되어 있습니다.

## 6. 일반 사용자에게 Docker 사용 권한 부여

현재 사용자를 `docker` 그룹에 추가합니다.

```bash
sudo usermod -aG docker "$USER"
```

현재 터미널 세션에 변경된 그룹 권한을 적용합니다.

```bash
newgrp docker
```

> [!IMPORTANT]
> `newgrp docker`를 사용하지 않는 경우 로그아웃 후 다시 로그인하거나 Jetson을 재부팅해야 새 그룹 권한이 적용됩니다.

### 보안 참고 사항

`docker` 그룹에 속한 사용자는 호스트 시스템에서 사실상 관리자 수준의 작업을 수행할 수 있습니다. 신뢰할 수 있는 사용자 계정에만 이 권한을 부여해야 합니다.

## 7. 권한 적용 확인

이제 `sudo` 없이 Docker 정보를 조회합니다.

```bash
docker info | grep -i runtime
```

다음과 같이 출력되면 권한 설정과 NVIDIA Runtime 등록 상태가 모두 정상입니다.

```text
Runtimes: io.containerd.runc.v2 nvidia runc
Default Runtime: runc
```

## 8. `Default Runtime: runc`의 의미

기본 Runtime이 `runc`로 표시되는 것은 문제가 아닙니다.

```text
Default Runtime: runc
```

일반 컨테이너는 기본적으로 `runc`를 사용하고, GPU가 필요한 컨테이너를 실행할 때 NVIDIA Runtime을 명시하면 됩니다.

```bash
docker run --runtime nvidia [기타 옵션] 이미지이름
```

예를 들어 이후 LLM 또는 GPU 추론 컨테이너를 실행할 때 `--runtime nvidia` 옵션을 사용합니다.

> [!NOTE]
> 사용할 컨테이너 이미지의 안내에 따라 `--runtime nvidia`, GPU 관련 환경 변수, 장치 마운트 등의 추가 옵션이 필요할 수 있습니다.

## 9. Conda 환경과 Docker의 관계

터미널 프롬프트에 Conda 기본 환경이 표시되어 있어도 Docker 실행에는 문제가 없습니다.

```text
(base) hansungai@ubuntu:~$
```

Conda 환경과 Docker 컨테이너는 서로 독립적으로 동작합니다.

```text
Jetson Host
├─ Conda 환경
│  ├─ base
│  └─ ~/venvs/jetson 또는 기타 Python 환경
│
└─ Docker 환경
   ├─ 독립된 파일 시스템
   ├─ 컨테이너 내부 패키지
   └─ NVIDIA Runtime을 통한 Jetson GPU 접근
```

따라서 `(base)`가 활성화된 상태에서도 `docker run` 명령을 사용할 수 있습니다. 컨테이너 내부의 Python과 라이브러리는 호스트의 Conda 환경과 별도로 관리됩니다.

## 10. 현재 단계의 결론

현재까지 완료된 작업은 다음과 같습니다.

- Docker가 이미 설치되어 있음을 확인했습니다.
- NVIDIA Container Toolkit이 이미 설치되어 있음을 확인했습니다.
- Docker에 `nvidia` Runtime이 등록되어 있음을 확인했습니다.
- 현재 사용자를 `docker` 그룹에 추가했습니다.
- 일반 사용자 계정에서 `sudo` 없이 Docker 명령을 실행할 수 있게 했습니다.
- Conda 환경과 Docker 환경이 서로 독립적임을 확인했습니다.

즉, 이 단계는 **Docker/NVIDIA 환경 신규 설치**가 아니라 **기존 환경 점검 및 Docker 사용 권한 설정** 단계입니다.

## 11. 다음 단계

다음 단계에서는 목적에 맞는 Jetson 호환 컨테이너 이미지를 준비하고 NVIDIA Runtime으로 실행합니다.

```text
기존 Docker/NVIDIA 환경 확인
    ↓
Docker 사용자 권한 설정
    ↓
Jetson 호환 이미지 확인 또는 다운로드
    ↓
NVIDIA Runtime으로 컨테이너 실행
    ↓
컨테이너 내부 GPU/CUDA 동작 검증
    ↓
LLM 또는 AI 애플리케이션 실행
```

## 빠른 점검표

- [x] `docker --version`으로 Docker 설치 확인
- [x] `nvidia-ctk --version`으로 NVIDIA Container Toolkit 확인
- [x] Docker의 `nvidia` Runtime 등록 상태 확인
- [x] 현재 사용자를 `docker` 그룹에 추가
- [x] `sudo` 없이 Docker 정보 조회 확인
- [x] Conda와 Docker가 별개 환경임을 확인
- [ ] Jetson 호환 컨테이너 이미지 준비
- [ ] NVIDIA Runtime으로 컨테이너 실행
- [ ] 컨테이너 내부 GPU 사용 여부 확인
