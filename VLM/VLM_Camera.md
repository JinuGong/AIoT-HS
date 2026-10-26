# Jetson VLM Camera 실시간 추론 실습

## 1. 실습 목표

Jetson Orin Nano에서 소규모 VLM을 실행하고, 입력 이미지 및 실행 구조를 최적화한 뒤 USB 카메라와 연결하여 실시간 영상 위에서 비동기적으로 VLM 추론을 수행한다.

최종 목표는 다음과 같다.

```text
USB Camera
    ↓
실시간 Frame Capture
    ↓
Latest Frame Buffer
    ↓
320px Resize
    ↓
SmolVLM2
    ↓
Semantic Caption
    ↓
PC Browser / Terminal
```

핵심 아이디어는 카메라의 모든 프레임을 VLM에 넣는 것이 아니라, 카메라는 실시간으로 계속 동작시키고 VLM은 가장 최신 프레임만 비동기적으로 분석하는 것이다.

---

## 2. 실습 환경

### Hardware

- NVIDIA Jetson Orin Nano Super Developer Kit
- Memory: 8 GB Unified Memory
- USB Camera

### Software

- Jetson Linux R36.4.7
- CUDA 12.6
- llama.cpp
- Python 3.10 virtual environment
- OpenCV
- Flask
- requests

### Model

Language model:

```text
SmolVLM2-2.2B-Instruct-Q4_K_M.gguf
```

Multimodal projector:

```text
mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf
```

---

# 3. Jetson 성능 모드 설정

실험 전 Jetson을 최대 성능 모드로 설정한다.

```bash
sudo nvpmodel -m 2
sudo jetson_clocks
```

확인:

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

본 실험에서는 다음 조건에서 측정하였다.

```text
Power Mode : MAXN_SUPER
GPU Clock  : 약 1.02 GHz
EMC Clock  : 약 3.2 GHz
```

---

# 4. VLM 실행 환경

작업 디렉터리:

```bash
cd ~/vlm_lab
source vlm_env/bin/activate
```

llama.cpp 디렉터리:

```bash
cd ~/vlm_lab/llama.cpp
```

모델 경로:

```bash
MODEL_Q4=~/vlm_lab/models/smolvlm2/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf
MMPROJ=~/vlm_lab/models/smolvlm2/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf
```

---

# 5. 최적화 1: Multimodal Projector GPU Offload

처음에는 `--no-mmproj-offload`를 사용하여 multimodal projector를 CPU에서 실행할 수 있다.

하지만 Jetson Orin Nano에서 CPU 기반 vision processing은 매우 느렸다.

실험 결과:

```text
CPU mmproj : 약 58.6 s / request
GPU mmproj : 약 4~6 s 수준
```

따라서 카메라 기반 VLM에서는 mmproj를 GPU에 올리는 것이 매우 중요하다.

기본적으로 `--no-mmproj-offload`를 사용하지 않으면 mmproj가 GPU에서 실행된다.

---

# 6. 최적화 2: GPU Layer Offload

`-ngl`은 language model의 transformer layer를 GPU에 얼마나 offload할지 결정한다.

예:

```bash
-ngl 30
```

실험 결과:

| NGL | Cold latency |
|---:|---:|
| 0 | 6.24 s |
| 5 | 5.83 s |
| 10 | 5.31 s |
| 20 | 4.52 s |
| 25 | 4.12 s |
| 30 | 4.11 s |
| 40 | 4.13 s |
| 99 | 6.85 s |

약 `NGL=25~40` 구간에서 성능이 plateau를 보였으며, 본 실습에서는 대표값으로 다음을 사용하였다.

```bash
-ngl 30
```

---

# 7. 최적화 3: Persistent llama-server

`llama-mtmd-cli`를 매번 실행하면 다음 비용을 요청마다 다시 지불한다.

```text
Process startup
+ Model load
+ mmproj load
+ Vision encoding
+ Prompt evaluation
+ Decode
```

따라서 카메라 기반 애플리케이션에서는 모델을 메모리에 계속 유지하는 persistent server가 적합하다.

서버 실행:

```bash
cd ~/vlm_lab/llama.cpp

MODEL_Q4=~/vlm_lab/models/smolvlm2/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf
MMPROJ=~/vlm_lab/models/smolvlm2/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf

./build/bin/llama-server \
  -m "$MODEL_Q4" \
  --mmproj "$MMPROJ" \
  -ngl 30 \
  -c 512 \
  -np 1 \
  --cache-prompt \
  --host 127.0.0.1 \
  --port 8080
```

정상 실행 시:

```text
llama_server: model loaded
llama_server: listening on http://127.0.0.1:8080
```

---

# 8. Persistent Server 성능

Cold process:

```text
약 4.1 s
```

Persistent server:

```text
약 2.8~3.0 s / request
```

예시:

```text
prompt eval time ≈ 2.0 s
eval time        ≈ 0.76 s
total time       ≈ 2.8 s
decode speed     ≈ 40 token/s
```

모델과 mmproj를 매 요청마다 다시 로드하지 않기 때문에 실제 애플리케이션 latency가 감소한다.

---

# 9. Quantization 비교: Q4_K_M vs Q8_0

동일 조건에서 language model quantization만 변경하여 비교하였다.

| 항목 | Q4_K_M | Q8_0 |
|---|---:|---:|
| Prompt throughput | 196.89 tok/s | 198.89 tok/s |
| Decode throughput | **39.96 tok/s** | **33.92 tok/s** |
| Prompt latency | 2199 ms | 2177 ms |

Decode throughput 기준으로 Q4_K_M이 약 18% 빠르게 나타났다.

\[
\frac{39.96}{33.92}\approx1.18
\]

LLM autoregressive decode는 weight를 반복적으로 읽기 때문에 memory bandwidth의 영향을 크게 받는다.

따라서 weight size가 더 작은 Q4_K_M은 memory traffic 감소 측면에서 유리하다.

본 카메라 실습에서는 다음을 사용한다.

```text
Language Model : Q4_K_M
mmproj          : Q8_0
```

---

# 10. Jetson Unified Memory 주의사항

VLM을 반복 실행하던 중 다음 오류가 발생할 수 있다.

```text
cudaMalloc failed: out of memory
NvMapMemAllocInternalTagged error 12
```

프로세스가 남아 있는지 확인:

```bash
ps aux | grep -E "llama|mtmd" | grep -v grep
```

메모리 확인:

```bash
free -h
sudo tegrastats
```

필요한 경우 page cache를 정리할 수 있다.

```bash
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

실험 당시:

```text
Before
free        764 MiB
buff/cache  4.4 GiB

After
free        5.8 GiB
buff/cache  335 MiB
```

이후 동일한 mmproj GPU allocation이 정상적으로 수행되었다.

> `drop_caches`는 일반적인 애플리케이션 운용에서 매번 실행하는 명령이 아니라, 반복적인 실험 과정에서 memory state를 정리하기 위한 용도로 사용한다.

---

# 11. 최적화 4: 입력 이미지 해상도 감소

카메라 입력을 그대로 VLM에 전달하지 않고 resize하여 vision token 및 prompt 처리량을 줄인다.

실험:

```text
Original
1280
640
320
```

결과:

| Input | Prompt tokens | Prompt eval | Total |
|---|---:|---:|---:|
| Original | 433 | 2172 ms | 2945 ms |
| 1280급 | 421 | 1985 ms | 2746 ms |
| 640급 | 421 | 1987 ms | 2748 ms |
| 320급 | **171** | **795 ms** | **1326 ms** |

320px 입력에서 prompt token이 크게 감소하면서 전체 latency가 약 절반 이하로 감소하였다.

\[
2.95\text{ s}\rightarrow1.33\text{ s}
\]

따라서 카메라 기반 VLM에서는 다음을 사용한다.

```text
VLM input max size = 320 px
```

---

# 12. 카메라 구조

카메라는 약 30 FPS로 동작하지만 VLM은 약 1초 이상이 필요하다.

모든 프레임을 queue에 넣으면 latency가 계속 누적된다.

잘못된 구조:

```text
Frame 1 → VLM
Frame 2 → Queue
Frame 3 → Queue
...
Frame 90 → Queue
```

3초 뒤 분석 결과가 이미 과거 상황을 설명하게 된다.

따라서 latest-frame 구조를 사용한다.

```text
Camera
  ↓
latest_frame

VLM busy
  ↓
새로운 frame은 latest_frame을 계속 덮어씀

VLM 완료
  ↓
현재 가장 최신 frame 분석
```

즉 오래된 frame을 의도적으로 버린다.

---

# 13. Headless Camera + VLM

Jetson에 모니터가 연결되어 있지 않다면 `cv2.imshow()`를 사용할 수 없다.

SSH / VS Code Remote 환경에서는 다음과 같은 오류가 발생할 수 있다.

```text
qt.qpa.xcb: could not connect to display
Can't initialize GTK backend
```

따라서 headless 환경에서는 화면 출력 없이 camera capture와 VLM inference를 수행한다.

필요 패키지:

```bash
cd ~/vlm_lab
source vlm_env/bin/activate

pip install opencv-python requests
```

실행:

```bash
python ~/vlm_lab/camera_vlm_headless.py
```

실제 결과:

```text
Camera opened.
Press Ctrl+C to stop.

[VLM] 1.41s | 320x240 | A man with glasses is looking at the camera...
[VLM] 1.25s | 320x240 | A man with glasses is looking at the camera...
[Camera] FPS = 19.4

[VLM] 1.24s | 320x240 | A man with glasses is looking at the camera...
[VLM] 1.25s | 320x240 | A person with glasses is looking at the camera...
[Camera] FPS = 29.4

[VLM] 1.15s | 320x240 | A young man with glasses is looking up...
[Camera] FPS = 29.8
```

카메라 capture는 warm-up 이후 약 30 FPS 수준으로 동작하였다.

---

# 14. llama-server 측정 결과

320px 카메라 프레임에서는 steady state에서 다음과 같은 결과를 얻었다.

```text
Prompt tokens     ≈ 171
Prompt eval       ≈ 0.79~0.80 s
Prompt throughput ≈ 214~215 token/s

Decode throughput ≈ 41 token/s
Decode latency    ≈ 0.32~0.48 s

Total latency     ≈ 1.11~1.28 s
```

대표적인 결과:

```text
prompt eval time = 795.02 ms / 171 tokens
eval time        = 315.24 ms / 14 tokens
total time       = 1110.26 ms
```

따라서 VLM semantic update rate는 대략:

\[
f_{\mathrm{VLM}}
\approx
\frac{1}{1.2}
\approx
0.83\text{ Hz}
\]

정도이다.

---

# 15. 실시간의 의미

현재 시스템은 VLM이 카메라의 모든 frame을 처리하는 구조가 아니다.

카메라:

```text
약 30 FPS
```

VLM:

```text
약 0.8 Hz
```

따라서 약 35~40개의 카메라 frame마다 한 번씩 최신 장면을 의미적으로 해석한다고 볼 수 있다.

```text
Camera frame
1  2  3  4  ... 35 36 37 ...
↓
VLM
1                 36
```

이 구조에서는 영상 자체는 실시간으로 유지되고, semantic description만 비동기적으로 갱신된다.

---

# 16. Web Camera + VLM

Headless Jetson에서 카메라 영상을 직접 띄우는 대신 Jetson을 웹 서버로 사용하고 PC 브라우저에서 확인할 수 있다.

구조:

```text
PC Browser
    │
    │ HTTP :5000
    ▼
Jetson Flask
    │
    ├── MJPEG Camera Stream
    │
    └── Latest VLM Caption
             │
             ▼
        llama-server
        127.0.0.1:8080
```

llama-server는 외부에 노출하지 않고 Jetson 내부에서만 접근한다.

```text
127.0.0.1:8080
```

웹 서버만 외부에서 접근 가능하도록 한다.

```text
0.0.0.0:5000
```

---

# 17. Flask 설치

```bash
cd ~/vlm_lab
source vlm_env/bin/activate

pip install flask
```

---

# 18. Web Camera 실행

llama-server를 먼저 실행한다.

Terminal 1:

```bash
cd ~/vlm_lab/llama.cpp

MODEL_Q4=~/vlm_lab/models/smolvlm2/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf
MMPROJ=~/vlm_lab/models/smolvlm2/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf

./build/bin/llama-server \
  -m "$MODEL_Q4" \
  --mmproj "$MMPROJ" \
  -ngl 30 \
  -c 512 \
  -np 1 \
  --cache-prompt \
  --host 127.0.0.1 \
  --port 8080
```

Terminal 2:

```bash
cd ~/vlm_lab
source vlm_env/bin/activate

python web_camera_vlm.py
```

Jetson IP 확인:

```bash
hostname -I
```

예:

```text
192.168.0.25
```

PC 브라우저에서:

```text
http://192.168.0.25:5000
```

으로 접속한다.

---

# 19. Web UI 구성

브라우저에서는 다음 정보를 표시할 수 있다.

```text
Jetson Camera + VLM

┌─────────────────────────────┐
│                             │
│      Real-time Camera       │
│                             │
└─────────────────────────────┘

A person with glasses is
looking at the camera.

VLM latency: 1.21 s
```

영상은 MJPEG stream으로 지속적으로 갱신되고, VLM caption은 별도의 API를 통해 주기적으로 업데이트한다.

---

# 20. 전체 최적화 결과

본 실습의 최적화 흐름은 다음과 같다.

```text
CPU mmproj
≈ 58.6 s
    ↓
GPU mmproj
≈ 6 s
    ↓
NGL ≈ 30
≈ 4.1 s cold
    ↓
Persistent llama-server
≈ 2.8~3.0 s
    ↓
320px input
≈ 1.3 s
    ↓
Camera latest-frame
≈ 1.1~1.3 s steady state
```

최종 조건:

```text
Jetson Orin Nano Super
MAXN_SUPER

SmolVLM2-2.2B-Instruct
Q4_K_M Language Model
Q8_0 mmproj

NGL = 30
Context = 512
Server slots = 1

mmproj GPU offload
Persistent llama-server

Camera ≈ 30 FPS
VLM input = 320 px
VLM ≈ 0.8 Hz
```

---

# 21. Roofline 관점에서의 해석

앞선 Roofline 실험과 연결하면 각 최적화의 의미를 다음과 같이 정리할 수 있다.

### Quantization

```text
Q8 → Q4
```

weight size 및 memory traffic 감소.

특히 autoregressive decode에서 효과가 나타났다.

```text
Q8 decode ≈ 33.9 token/s
Q4 decode ≈ 40.0 token/s
```

### GPU Offload

CPU에서 처리하던 vision/mmproj 연산을 GPU로 이동하여 연산 성능을 크게 개선한다.

```text
CPU mmproj ≈ 58.6 s
GPU mmproj ≈ 수 초 수준
```

### Image Resize

입력 이미지 token 수를 줄여 vision/prompt processing 계산량을 감소시킨다.

```text
433 tokens
   ↓
171 tokens
```

이에 따라 prompt processing latency도 감소한다.

```text
2.17 s
   ↓
0.80 s
```

### Persistent Server

모델을 매 요청마다 다시 로드하지 않는다.

```text
Load Model
Load mmproj
```

비용을 initialization 단계에서 한 번만 지불한다.

### Latest-frame Architecture

이는 kernel-level optimization이 아니라 system-level optimization이다.

처리할 수 없는 frame을 queue에 쌓는 대신 버림으로써 latency accumulation을 방지한다.

---

# 22. 최종 결론

Jetson Orin Nano에서 SmolVLM2를 카메라의 모든 frame에 적용하는 frame-by-frame real-time VLM은 현실적으로 어렵다.

하지만 다음 구조는 충분히 가능하다.

```text
Camera
≈ 30 FPS
   ↓
Real-time Capture
   ↓
Latest Frame
   ↓
VLM
≈ 1.1~1.3 s
   ↓
Semantic Description
≈ 0.8 Hz
```

즉,

> 카메라는 실시간으로 동작시키고 VLM은 비동기 semantic perception layer로 사용하는 방식

이 현실적인 Edge VLM 시스템 구조이다.

향후에는 다음과 같은 event-triggered 구조로 확장할 수 있다.

```text
Camera
   ↓
YOLO / Motion Detection
   ↓
Event?
 ┌─┴─┐
No  Yes
│    ↓
│   VLM
│    ↓
│ Semantic Reasoning
│
└→ Continue
```

이 경우 VLM을 항상 실행하지 않고 실제 의미 분석이 필요한 순간에만 호출하여 전력과 계산량을 추가로 줄일 수 있다.
