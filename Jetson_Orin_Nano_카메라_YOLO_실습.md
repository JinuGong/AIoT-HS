# Jetson Orin Nano 카메라 및 YOLO 실습

이 문서는 Jetson Orin Nano에서 시스템 상태와 카메라를 확인하고, Flask로 실시간 영상을 전송한 뒤 YOLO11 객체 탐지를 실행하는 과정을 정리한 실습 가이드입니다. PyTorch와 CUDA 버전이 맞지 않아 YOLO가 CPU로 실행되는 경우, ONNX와 TensorRT를 이용해 GPU 추론 엔진을 만드는 방법도 함께 설명합니다.

> [!NOTE]
> 명령은 특별한 안내가 없는 한 Jetson 터미널에서 실행합니다.

## 전체 실습 흐름

```text
Jetson 상태 확인
    ↓
네트워크 및 Python 환경 준비
    ↓
OpenCV 카메라 확인
    ↓
Flask 카메라 스트리밍
    ↓
YOLO11 객체 탐지
    ↓
PyTorch CUDA 동작 확인
    ↓
필요한 경우 ONNX → TensorRT 변환
```

## 1. GPU 및 CPU 상태 확인

Jetson의 CPU, GPU, 메모리 사용량과 온도 등을 실시간으로 확인합니다.

```bash
tegrastats
```

종료하려면 `Ctrl+C`를 누릅니다.

## 2. Wi-Fi 연결(권장하지 않음)

다음 예시는 PEAP/GTC 방식의 엔터프라이즈 Wi-Fi 연결을 생성합니다.

> [!WARNING]
> 아래 설정은 서버 인증서를 검증하지 않도록 구성하므로 보안상 권장되지 않습니다. 가능하면 관리자에게 CA 인증서와 공식 연결 방법을 확인하세요.

### 2.1 연결 프로필 생성

`본인 id`를 실제 계정 ID로 변경합니다. 인터페이스 이름 `wlP1p1s0`도 `nmcli device status`에서 확인한 값과 다르면 수정합니다.

```bash
sudo nmcli connection add \
  type wifi \
  ifname wlP1p1s0 \
  con-name "hansung-GTC" \
  ssid "hansung" \
  -- \
  wifi-sec.key-mgmt wpa-eap \
  802-1x.eap peap \
  802-1x.phase2-auth gtc \
  802-1x.identity "본인 id"
```

### 2.2 시스템 CA 인증서 사용 해제

```bash
sudo nmcli connection modify "hansung-GTC" 802-1x.system-ca-certs no
```

### 2.3 연결 시작

```bash
sudo nmcli --ask connection up "hansung-GTC"
```

암호 입력을 요구하면 해당 계정의 암호를 입력합니다.

### 2.4 자동 연결 설정

다음 부팅 때 자동으로 연결되지 않는 경우 실행합니다.

```bash
sudo nmcli connection modify "hansung-GTC" connection.autoconnect yes
```

## 3. Python 환경 확인

Jetson에 설치된 Python 3 버전을 확인합니다.

```bash
python3 --version
```

## 4. OpenCV 설치 및 확인

### 4.1 OpenCV 설치

```bash
sudo apt update
sudo apt install -y python3-opencv
```

### 4.2 설치 확인

```bash
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
```

### 4.3 Conda와 시스템 OpenCV가 충돌하는 경우

Conda 환경과 `apt`로 설치한 OpenCV 사이에 버전 또는 경로 문제가 있으면 Conda를 비활성화하고, 시스템 패키지를 사용할 수 있는 가상환경을 생성합니다.

```bash
conda deactivate
/usr/bin/python3 -m venv --system-site-packages ~/venvs/jetson
source ~/venvs/jetson/bin/activate
```

이후 실습 명령은 가상환경이 활성화된 상태에서 실행합니다.

## 5. 카메라 장치 확인

USB 카메라가 연결되었는지 확인합니다.

```bash
ls /dev/video*
```

예를 들어 `/dev/video0`이 보이면 OpenCV에서 일반적으로 장치 번호 `0`으로 접근할 수 있습니다.

## 6. Flask 기반 실시간 카메라 스트리밍

### 6.1 Flask 설치

```bash
pip install flask
```

### 6.2 `camera_stream.py` 생성

```bash
cat > camera_stream.py <<'PY'
from flask import Flask, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        encoded, buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <html>
    <body>
        <h2>Jetson Orin Nano - Live Camera</h2>
        <img src="/video_feed" width="800">
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
PY
```

### 6.3 서버 실행

```bash
python3 camera_stream.py
```

Jetson과 연결된 Windows PC의 브라우저에서 다음 주소에 접속합니다.

```text
http://192.168.55.1:5000
```

> [!TIP]
> `192.168.55.1`은 Jetson의 USB 네트워크 연결에서 흔히 사용하는 주소입니다. 접속되지 않으면 Jetson에서 `ip addr`를 실행해 실제 IP 주소를 확인하세요.

## 7. YOLO11 설치 및 기본 실행

### 7.1 Ultralytics 설치

```bash
pip install ultralytics
```

### 7.2 설치 환경 확인

```bash
yolo checks
```

### 7.3 카메라 영상으로 YOLO 실행

GUI가 없는 환경에서는 다음과 같이 실행할 수 있습니다.

```bash
yolo predict model=yolo11n.pt source=0
```

최초 실행 시 모델 파일이 자동으로 내려받아질 수 있습니다.

## 8. Flask 기반 YOLO 실시간 스트리밍

### 8.1 `yolo_stream.py` 생성

```bash
cat > yolo_stream.py <<'PY'
from flask import Flask, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# 가장 작은 YOLO11 모델
model = YOLO("yolo11n.pt")

# USB 카메라
cap = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO 추론
        results = model(frame, verbose=False)

        # 경계 상자와 레이블 표시
        annotated_frame = results[0].plot()

        # JPEG 인코딩
        encoded, buffer = cv2.imencode(".jpg", annotated_frame)
        if not encoded:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <html>
    <head>
        <title>Jetson YOLO</title>
    </head>
    <body>
        <h2>Jetson Orin Nano - YOLO11 Live Detection</h2>
        <img src="/video_feed" width="800">
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
PY
```

### 8.2 실행

```bash
python3 yolo_stream.py
```

Windows PC의 브라우저에서 다음 주소에 접속합니다.

```text
http://192.168.55.1:5000
```

## 9. YOLO가 느릴 때: PyTorch의 CUDA 사용 여부 확인

다음 명령으로 PyTorch 버전과 CUDA 사용 가능 여부를 확인합니다.

```bash
python3 -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Jetson GPU를 PyTorch에서 정상적으로 사용할 수 있다면 다음과 같이 출력되어야 합니다.

```text
CUDA: True
```

다음과 같이 출력되면 PyTorch가 GPU를 사용하지 못하고 CPU에서 추론하고 있다는 뜻입니다.

```text
Torch: 2.14.0+cu130
CUDA: False
```

## 10. CUDA 버전 불일치 진단

### 10.1 Jetson의 CUDA 버전 확인

```bash
nvcc --version
```

예를 들어 Jetson의 CUDA가 12.6인데 설치된 PyTorch가 `+cu130`, 즉 CUDA 13.0용 빌드라면 서로 호환되지 않을 수 있습니다.

```text
Jetson CUDA        : 12.6
PyTorch CUDA build : 13.0
```

이 경우 다음과 같은 증상이 나타날 수 있습니다.

- NVIDIA 드라이버가 너무 오래되었다는 경고
- CUDA 초기화 실패
- `torch.cuda.is_available()` 결과가 `False`
- YOLO가 CPU에서 실행되어 추론 속도가 매우 느림

### 10.2 Jetson의 L4T, CUDA 및 TensorRT 상태 확인

L4T 버전 확인:

```bash
cat /etc/nv_tegra_release
```

CUDA 버전 확인:

```bash
nvcc --version
```

TensorRT 패키지 확인:

```bash
dpkg -l | grep -i tensorrt
```

TensorRT 실행 파일 확인:

```bash
/usr/src/tensorrt/bin/trtexec --version
```

실습 환경에서 확인된 값의 예시는 다음과 같습니다.

```text
L4T      : R36.4.7
CUDA     : 12.6
TensorRT : 10.3.0
```

이 경우 Jetson의 CUDA와 TensorRT는 정상이지만, `pip`로 설치한 PyTorch가 Jetson 환경과 맞지 않을 가능성이 있습니다.

## 11. 잘못 설치된 PyTorch 제거

호환되지 않는 PyTorch 패키지를 제거합니다.

```bash
pip uninstall -y torch torchvision torchaudio
```

> [!CAUTION]
> Jetson에서는 일반 PC와 달리 `pip install torch`로 받은 패키지가 현재 JetPack/CUDA 환경과 호환되지 않을 수 있습니다. PyTorch GPU 실행이 꼭 필요하다면 사용 중인 JetPack 버전에 맞는 NVIDIA 제공 패키지를 사용해야 합니다.

## 12. CPU PyTorch로 YOLO 모델을 ONNX로 변환

GPU 추론이 아니라 모델 변환만 수행하기 위해 CPU용 PyTorch를 설치합니다.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

설치 결과를 확인합니다.

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CPU 전용 PyTorch이므로 `torch.cuda.is_available()`이 `False`여도 정상입니다.

YOLO11n 모델을 ONNX 형식으로 내보냅니다.

```bash
yolo export model=yolo11n.pt format=onnx imgsz=640 simplify=True
```

성공하면 현재 디렉터리에 `yolo11n.onnx`가 생성됩니다.

```text
YOLO11n(.pt)
    ↓
CPU PyTorch로 모델 변환
    ↓
ONNX(.onnx)
```

## 13. ONNX 모델을 TensorRT FP16 엔진으로 변환

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolo11n.onnx \
  --saveEngine=yolo11n_fp16.engine \
  --fp16
```

TensorRT는 Jetson GPU에 맞는 커널과 tactic을 탐색하여 최적화된 엔진을 생성합니다. 성공하면 마지막 부분에 다음 메시지가 표시됩니다.

```text
&&&& PASSED TensorRT.trtexec
```

실습 환경의 측정 예시는 다음과 같습니다.

```text
GPU Compute Time : 약 6.71 ms
Host Latency     : 약 7.40 ms
Throughput       : 약 148.6 qps
```

모델 추론 시간만 기준으로 하면 다음과 같이 약 149 FPS에 해당합니다.

```text
1 / 0.0067 ≈ 149 FPS
```

> [!IMPORTANT]
> 이 수치는 TensorRT 모델 추론 자체의 벤치마크입니다. 카메라 캡처, 전처리, 후처리, JPEG 인코딩 및 네트워크 전송 시간이 포함된 실제 스트리밍 FPS와는 다릅니다.

## 14. TensorRT를 사용하는 이유

PyTorch와 CUDA가 호환되지 않는 경우의 실행 구조는 다음과 같습니다.

```text
Camera
  ↓
OpenCV
  ↓
Ultralytics
  ↓
PyTorch
  ↓
CUDA 버전 불일치
  ↓
CPU 추론
  ↓
느린 처리 속도
```

TensorRT 엔진을 사용하면 다음과 같이 PyTorch GPU 문제를 우회하고 Jetson GPU에서 직접 추론할 수 있습니다.

```text
Camera
  ↓
OpenCV
  ↓
TensorRT Engine
  ↓
Jetson GPU
  ↓
빠른 추론
```

## 15. TensorRT Python 환경 확인

### 15.1 TensorRT Python 바인딩 확인

```bash
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

실습 환경의 출력 예시:

```text
10.3.0
```

### 15.2 CUDA Python 바인딩 확인

```bash
python3 -c "from cuda.bindings import runtime; print(runtime.cudaGetDeviceCount())"
```

정상 출력 예시:

```text
(<cudaError_t.cudaSuccess: 0>, 1)
```

마지막 값 `1`은 CUDA 장치가 한 개 인식되었다는 뜻입니다.

## 16. TensorRT 엔진의 입출력 확인

```bash
python3 - <<'PY'
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)

with open("yolo11n_fp16.engine", "rb") as engine_file:
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_file.read())

print("Num I/O tensors:", engine.num_io_tensors)

for index in range(engine.num_io_tensors):
    name = engine.get_tensor_name(index)
    print(
        index,
        name,
        engine.get_tensor_mode(name),
        engine.get_tensor_dtype(name),
        engine.get_tensor_shape(name),
    )
PY
```

YOLO11n의 입출력 형태 예시는 다음과 같습니다.

```text
Input  : (1, 3, 640, 640)
Output : (1, 84, 8400)
```

각 숫자의 의미는 다음과 같습니다.

| 값 | 의미 |
|---:|---|
| `1` | 배치 크기 |
| `3` | RGB 채널 수 |
| `640 × 640` | 입력 이미지 크기 |
| `84` | 경계 상자 좌표 4개 + COCO 클래스 80개 |
| `8400` | 후보 경계 상자 수 |

## 17. 최종 실시간 처리 구조

```text
USB Camera
    ↓
OpenCV
    ↓
Resize / Letterbox
    ↓
NumPy
    ↓
CUDA Host-to-Device Copy
    ↓
TensorRT FP16 Engine
    ↓
Jetson Orin GPU
    ↓
CUDA Device-to-Host Copy
    ↓
YOLO 후처리 및 NMS
    ↓
경계 상자 표시
    ↓
JPEG 인코딩
    ↓
Flask
    ↓
Windows 웹 브라우저
```

브라우저 접속 주소:

```text
http://192.168.55.1:5000
```

> [!NOTE]
> `trtexec`로 엔진을 생성하는 것만으로 Flask 프로그램이 자동으로 TensorRT 엔진을 사용하지는 않습니다. 실제 스트리밍에서는 `yolo11n_fp16.engine`의 입력 전처리, CUDA 메모리 복사, 추론 실행, 출력 후처리 및 NMS를 수행하는 별도의 TensorRT 추론 코드가 필요합니다.

## 18. Jetson 종료

모든 작업을 저장하고 다음 명령으로 Jetson을 안전하게 종료합니다.

```bash
sudo shutdown -h now
```

## 빠른 점검표

- [ ] `tegrastats`로 시스템 상태 확인
- [ ] 네트워크 연결 확인
- [ ] Python 3 및 OpenCV 확인
- [ ] `/dev/video*` 카메라 장치 확인
- [ ] Flask 카메라 스트리밍 확인
- [ ] Ultralytics 및 YOLO11 실행 확인
- [ ] `torch.cuda.is_available()` 확인
- [ ] 필요하면 YOLO 모델을 ONNX로 변환
- [ ] TensorRT FP16 엔진 생성 및 성능 확인
- [ ] 실제 TensorRT 스트리밍 코드를 연결해 브라우저에서 확인
- [ ] 실습 종료 후 Jetson 안전 종료
