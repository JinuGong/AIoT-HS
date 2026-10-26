# Jetson Orin Nano 경량 LLM 실행 실습

이 문서는 Jetson Orin Nano에서 NVIDIA Jetson용 `llama.cpp` Docker 이미지를 이용해 경량 LLM을 실행하고, Windows 브라우저에서 Web UI에 접속한 과정을 정리한 실습 노트입니다.

실습 결과 모델과 웹 서버는 정상적으로 실행되었지만, CUDA 호환성 문제로 GPU 가속은 활성화되지 않았습니다. 따라서 현재 결과는 **CPU 추론 기준**입니다.

> [!IMPORTANT]
> 실제 네트워크 주소는 보안을 위해 `{JETSON_USB_IP}`, `{WINDOWS_USB_IP}` 등의 자리표시자로 변경했습니다.

## 1. 실습 목표

앞선 단계에서 다음 환경을 준비했습니다.

```text
Jetson Orin Nano Super
├─ JetPack / L4T R36.4.7
├─ CUDA 12.6
├─ TensorRT 10.3
├─ Docker 29.1.3
├─ NVIDIA Container Toolkit 1.16.2
└─ 유선 인터넷 연결
```

이번 실습의 목표는 Docker를 이용하여 Jetson에서 실제 경량 LLM과 웹 서버를 실행하는 것입니다.

```text
Jetson Orin Nano
      │
      ├─ Docker
      │    └─ NVIDIA Jetson용 llama.cpp
      │         └─ Gemma 4 E2B
      │              └─ Q4_K_S GGUF
      │
      └─ llama-server
           └─ Web UI
```

## 2. NVIDIA Jetson용 `llama.cpp` 이미지 다운로드

NVIDIA에서 Jetson Orin용으로 제공하는 Docker 이미지를 다운로드합니다.

```bash
docker pull ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin
```

정상적으로 다운로드되면 다음과 같은 메시지가 출력됩니다.

```text
Digest: sha256:[DIGEST]
Status: Downloaded newer image for
ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin
```

### Docker 이미지와 LLM 모델의 차이

Docker 이미지와 LLM 모델은 서로 다른 구성 요소입니다.

| 구성 요소 | 역할 |
|---|---|
| Docker 이미지 | `llama.cpp`와 필요한 실행 환경 제공 |
| GGUF 모델 | 실제 LLM의 파라미터와 가중치 저장 |

따라서 Docker 이미지를 내려받은 뒤 실제 LLM 모델도 별도로 다운로드해야 합니다.

## 3. Gemma 4 E2B 실행

경량 LLM으로 Gemma 4 E2B instruction-tuned 모델을 사용했습니다. Jetson의 제한된 메모리를 고려하여 `Q4_K_S` GGUF 양자화 버전을 선택했습니다.

```bash
docker run -it --rm \
  --runtime=nvidia \
  --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin \
  llama-server \
  -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S
```

### Docker 옵션 설명

| 옵션 | 의미 |
|---|---|
| `-it` | 대화형 터미널로 컨테이너 실행 |
| `--rm` | 컨테이너 종료 후 해당 컨테이너 자동 삭제 |
| `--runtime=nvidia` | 컨테이너에서 NVIDIA GPU Runtime 사용 |
| `--network host` | 컨테이너가 Jetson 호스트의 네트워크를 직접 사용 |
| `-v ...` | Hugging Face 모델 캐시를 Jetson 호스트에 보존 |

모델 캐시 마운트는 다음과 같이 연결됩니다.

```text
Jetson host
$HOME/.cache/huggingface
          ↕
Docker container
/root/.cache/huggingface
```

`--rm` 옵션 때문에 컨테이너는 종료 시 삭제되지만, 모델 파일은 Jetson의 `$HOME/.cache/huggingface`에 남습니다. 따라서 이후 실행에서는 같은 모델을 다시 다운로드할 필요가 없습니다.

## 4. GGUF와 `Q4_K_S` 양자화

이번 실습에서는 모델을 다음 형태로 사용했습니다.

```text
Gemma 4 E2B
    ↓
GGUF
    ↓
Q4_K_S
```

GGUF는 `llama.cpp` 계열 도구에서 널리 사용하는 모델 파일 형식입니다. 모델의 가중치와 추론에 필요한 메타데이터를 저장합니다.

`Q4_K_S`는 모델 가중치를 대략 4비트 수준으로 표현하는 양자화 방식입니다.

```text
FP16 모델
약 16 bit / weight
        ↓
양자화
        ↓
Q4_K_S
약 4 bit / weight
```

양자화를 사용하면 모델의 저장 공간과 실행 시 메모리 사용량을 크게 줄일 수 있으므로, 메모리가 제한된 Edge 장치에서 LLM을 실행할 때 유용합니다. 다만 양자화 수준에 따라 모델 품질과 실행 성능이 달라질 수 있습니다.

## 5. 최초 모델 다운로드

최초 실행 시 Docker 컨테이너가 실제 GGUF 모델 파일을 다운로드했습니다.

```text
Downloading mmproj-BF16.gguf
Downloading gemma-4-E2B-it-Q4_K_S.gguf
```

다운로드 후 다음과 같은 로그가 출력되었습니다.

```text
load_model: loading model
...
model loaded
```

이는 모델 파일 다운로드와 메모리 로딩이 정상적으로 완료되었다는 뜻입니다.

## 6. `llama-server` 실행 상태 이해

모델 로딩 후 다음 메시지가 출력되었습니다.

```text
model loaded
listening on http://[LOOPBACK_ADDRESS]:8080
```

이 상태에서 터미널 출력이 멈춘 것처럼 보여도 정상입니다. 서버가 종료된 것이 아니라 클라이언트의 요청을 기다리고 있습니다.

```text
LLM 로드
   ↓
서버 시작
   ↓
8080 포트 listen
   ↓
사용자 요청 대기
```

## 7. 외부 PC에서 접속할 수 있도록 서버 공개

기본 설정에서는 서버가 loopback 주소에만 bind되어 Jetson 내부에서만 접근할 수 있었습니다.

Windows PC에서 접속하려면 실행 중인 서버를 `Ctrl+C`로 종료한 뒤, `--host 0.0.0.0`과 `--port 8080`을 추가하여 다시 실행합니다.

```bash
docker run -it --rm \
  --runtime=nvidia \
  --network host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin \
  llama-server \
  -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S \
  --host 0.0.0.0 \
  --port 8080
```

핵심 옵션은 다음과 같습니다.

| 옵션 | 의미 |
|---|---|
| `--host 0.0.0.0` | Jetson의 외부 네트워크 인터페이스에서도 서버 접근 허용 |
| `--port 8080` | 서버가 사용할 TCP 포트 지정 |

> [!WARNING]
> `0.0.0.0`에 bind하면 해당 포트에 도달할 수 있는 다른 장치에서도 서버에 접근할 수 있습니다. 신뢰할 수 있는 실습망에서만 사용하고, 공용 인터넷에 직접 노출하지 마세요.

## 8. Windows에서 Jetson LLM 접속

Windows PC와 Jetson은 USB-C 네트워크로 연결되어 있습니다.

```text
Windows PC
{WINDOWS_USB_IP}
       │
       │ USB-C
       ▼
Jetson Orin Nano
{JETSON_USB_IP}
```

Windows 브라우저에서 다음 주소에 접속합니다.

```text
http://{JETSON_USB_IP}:8080
```

`llama.cpp` Web UI가 표시되면 브라우저에서 Gemma 모델과 대화할 수 있습니다.

```text
Windows Browser
       │
       │ HTTP
       ▼
{JETSON_USB_IP}:8080
       │
       ▼
llama-server
       │
       ▼
Gemma 4 E2B Q4_K_S
```

## 9. Jetson 자원 사용량 확인

LLM 추론 중 별도의 Jetson 터미널에서 다음 명령을 실행합니다.

```bash
tegrastats
```

주요 관찰 항목은 다음과 같습니다.

| 항목 | 의미 |
|---|---|
| `RAM` | 시스템 메모리 사용량 |
| `CPU` | CPU 코어별 사용률 |
| `GR3D_FREQ` | GPU 사용 상태 및 주파수 |
| `VDD_IN` | 보드 전체 입력 전력 |
| `VDD_CPU_GPU_CV` | CPU, GPU 및 관련 연산 블록의 전력 |
| Temperature | CPU, GPU 및 SoC 온도 |

실제 추론 중 관찰된 값은 대략 다음과 같습니다.

| 항목 | 관찰값 |
|---|---:|
| RAM | 약 3.25 GB / 7.62 GB |
| CPU | 약 80~100% |
| `GR3D_FREQ` | 0% |
| `VDD_IN` | 약 8.4~8.7 W |
| 생성 속도 | 약 7.44 tokens/s |

> [!NOTE]
> 위 값은 해당 장비와 실행 시점에서 관찰한 예시입니다. 전력 모드, 온도, 컨텍스트 길이, 프롬프트, 모델 설정 및 백그라운드 작업에 따라 달라질 수 있습니다.

## 10. 문제 발견: GPU가 사용되지 않음

Docker 실행 초기에 다음 오류가 발생했습니다.

```text
ggml_cuda_init: failed to initialize CUDA:
CUDA driver version is insufficient for CUDA runtime version
```

`llama.cpp`가 CPU 추론을 지원하기 때문에 CUDA 초기화가 실패해도 모델과 서버는 정상적으로 실행될 수 있습니다.

실제 `tegrastats`에서도 다음 상태가 관찰되었습니다.

```text
CPU       → 높은 사용률
GR3D_FREQ → 0%
```

따라서 현재 추론 경로는 다음과 같이 판단할 수 있습니다.

```text
Gemma 4 E2B Q4_K_S
       ↓
llama.cpp
       ↓
CUDA 초기화
       ↓
실패
       ↓
CPU fallback
       ↓
ARM CPU 추론
```

즉, **LLM 실행에는 성공했지만 GPU 가속에는 아직 성공하지 않은 상태**입니다.

## 11. CPU 추론 결과

| 항목 | 결과 |
|---|---|
| 모델 | Gemma 4 E2B IT |
| 형식 | GGUF |
| 양자화 | Q4_K_S |
| RAM | 약 3.25 GB |
| GPU 사용률 | 0% |
| CPU 사용률 | 약 80~100% |
| 생성 속도 | 약 7.4 tokens/s |
| 입력 전력 | 약 8.4~8.7 W |
| 실행 방식 | CPU inference |

이번 실습을 통해 2B급 4비트 LLM을 Jetson의 ARM CPU만으로도 대화형으로 실행할 수 있음을 확인했습니다.

## 12. CUDA 호환성 문제

Jetson 호스트의 환경은 다음과 같습니다.

```text
L4T      : R36.4.7
CUDA     : 12.6
TensorRT : 10.3
```

그러나 사용한 Docker 이미지에서 CUDA 초기화 중 다음 오류가 발생했습니다.

```text
CUDA driver version is insufficient for CUDA runtime version
```

이는 현재 `latest-jetson-orin` 이미지의 CUDA Runtime 요구사항과 Jetson 호스트의 드라이버/L4T 스택 사이에 호환성 문제가 있을 가능성을 나타냅니다.

추가로 확인할 항목은 다음과 같습니다.

- Jetson의 정확한 JetPack 및 L4T 버전
- 컨테이너 이미지가 요구하는 L4T 및 CUDA 버전
- `latest-jetson-orin` 대신 호스트와 호환되는 고정 태그가 있는지 여부
- 컨테이너에서 CUDA 장치와 라이브러리가 정상적으로 노출되는지 여부
- 실제 GPU offload 레이어가 설정되어 있는지 여부

> [!IMPORTANT]
> LLM이 실행된다는 사실만으로 GPU를 사용한다고 판단하면 안 됩니다. 서버 로그와 `tegrastats`의 GPU 사용률을 함께 확인해야 합니다.

## 13. 현재 완성된 시스템

```text
                 Windows PC
                     │
                     │ Browser
                     │
          http://{JETSON_USB_IP}:8080
                     │
                     │ USB-C
                     ▼
          ┌──────────────────────┐
          │ Jetson Orin Nano     │
          │                      │
          │ Docker               │
          │   ↓                  │
          │ llama.cpp            │
          │   ↓                  │
          │ Gemma 4 E2B          │
          │ Q4_K_S / GGUF        │
          │   ↓                  │
          │ CPU inference        │
          └──────────────────────┘
                     │
                   eno1
                     │
                     ▼
                  Internet
```

현재까지 완료된 항목은 다음과 같습니다.

- Jetson용 `llama.cpp` Docker 이미지 다운로드
- GGUF 모델 다운로드 및 호스트 캐시 보존
- Gemma 4 E2B Q4_K_S 모델 로딩
- `llama-server` 실행
- Windows 브라우저에서 Web UI 접속
- CPU 기반 LLM 추론
- 처리 속도, RAM 및 전력 사용량 관찰
- CUDA 초기화 실패와 CPU fallback 확인

## 14. 다음 실습 단계

```text
현재 상태
CPU inference
약 7.4 tokens/s
       ↓
CUDA 호환성 확인 및 해결
       ↓
GPU offloading 활성화
       ↓
GPU inference
       ↓
CPU와 GPU의 성능·전력 비교
```

다음 단계에서는 같은 모델과 유사한 조건으로 CPU 및 GPU 실행 결과를 비교할 수 있습니다.

| 비교 항목 | CPU 실행 | GPU 실행 |
|---|---:|---:|
| 생성 속도(tokens/s) | 측정 완료 | 측정 예정 |
| CPU 사용률 | 측정 완료 | 측정 예정 |
| GPU 사용률 | 0% | 측정 예정 |
| RAM 사용량 | 측정 완료 | 측정 예정 |
| 입력 전력 | 측정 완료 | 측정 예정 |
| 온도 | 추가 기록 권장 | 추가 기록 권장 |

이를 통해 Edge AI 환경에서의 성능과 전력 소비 간 trade-off를 분석할 수 있습니다.

## 빠른 점검표

- [x] Jetson용 `llama.cpp` Docker 이미지 다운로드
- [x] Gemma 4 E2B Q4_K_S 모델 다운로드
- [x] Hugging Face 캐시를 호스트에 마운트
- [x] 모델 로딩 확인
- [x] `llama-server` 실행
- [x] 외부 접속을 위한 host 및 port 설정
- [x] Windows 브라우저에서 Web UI 확인
- [x] `tegrastats`로 자원 사용량 측정
- [x] CUDA 초기화 실패 확인
- [x] 현재 실행이 CPU inference임을 확인
- [ ] 호스트와 호환되는 컨테이너 이미지 태그 확인
- [ ] CUDA 장치 및 라이브러리 노출 상태 확인
- [ ] GPU offloading 활성화
- [ ] GPU 추론 성능 및 전력 측정
- [ ] CPU와 GPU 결과 비교
