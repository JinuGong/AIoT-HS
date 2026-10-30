# Jetson Orin Nano 이미지 분류 추론 성능 기준 실험

> **지능형 IoT 통신시스템 | 조별 사전 벤치마크 과제 | 총 14개 조**  
> **제출 장소:** e-Class → 해당 교과목 → **과제** 탭  
> **제출 대상:** 조별 **1회 제출** (조 대표 1명 제출)  
> **마감일:** e-Class 과제에 표시된 마감일을 따름

## 1. 실험 목적

최종 프로젝트에 앞서, 동일한 Jetson Orin Nano 보드 사이에 **기본 CNN GPU 추론 성능 편차가 있는지** 확인한다.

- 동일한 모델·입력·배치 크기·측정 프로그램을 모든 조에서 사용한다.
- **데이터셋 다운로드, 파일 읽기, 이미지 전처리 시간은 측정하지 않는다.**
- 이 실험은 **하드웨어 성능 점검용**이며, 조별 모델 정확도나 최종 프로젝트의 우열을 결정하는 실험이 아니다.
- 빠른 기록을 내기 위해 코드를 바꾸는 경쟁이 아니다. **동일 조건에서 정확하게 측정해 제출**하는 것이 목적이다.

## 2. 실험 조건 (모든 조 공통)

| 항목 | 설정 |
|---|---|
| 보드 | Jetson Orin Nano 8GB Super (JetPack 6.x, L4T R36.4.x, CUDA 12.6) |
| 전력 모드 | `MAXN_SUPER` |
| 클럭 | `jetson_clocks` 적용 |
| Python 환경 | `~/benchmark/venv` (Python 3.10, **torch 2.8.0 Jetson용 CUDA 휠**, `numpy<2`) |
| 모델 | 제공된 4-Conv CNN (무작위 초기화, 미학습) |
| 입력 형태 | `(B, 3, 224, 224)` |
| 입력 데이터 | 프로그램 내부에서 생성한 텐서 (GPU에 미리 적재) |
| 배치 크기 | **1, 16** 모두 측정 |
| Warm-up | 배치 크기별 **100회** |
| GPU 시간 측정 | 배치 크기별 **200회 × 3 trial** (CUDA Event) |
| 호스트 시간 측정 | 배치 크기별 **200회** 추가 측정 |
| 전체 프로그램 실행 | **각 조 3회** |
| 제출 방식 | **각 실행 결과 JSON 3개**를 하나의 ZIP으로 제출 |

위 측정 횟수는 코드에 이미 설정되어 있다. **`--warmup`이나 `--repeats` 값을 따로 지정하지 않는다.**  
따라서 학생은 **같은 명령을 출력 파일명만 바꾸어 세 번 실행**하면 된다.

> [!IMPORTANT]
> 모든 조가 **같은 torch 버전(2.8.0)**을 써야 한다. torch 버전이 다르면 보드 차이가 아니라 소프트웨어 차이를 재게 된다.

### 측정 지표

- `median_ms`: CUDA Event 기준 **GPU 스트림 경과시간**(ms/batch). 낮을수록 빠르다.
- `p95_ms`: GPU 스트림 경과시간의 95백분위수.
- `wall_median_ms`: CPU/Python 호출과 GPU 동기화 시간을 포함한 호스트 측 경과시간(ms/batch).
- `images_per_second`: `batch_size × 1000 / median_ms`로 계산한 **GPU 시간 기준 처리량**. 높을수록 빠르다.

**비교의 중심 지표는 B=1의 `median_ms`와 B=16의 `images_per_second`이다.**


## 3. 가상환경 설정 (조별 1회)

> [!IMPORTANT]
> 이 절의 명령은 **모두 sudo 없이** 실행한다. `sudo python`, `sudo pip`를 쓰면 가상환경이 아닌 시스템 Python이 사용되거나, 파일이 root 소유가 되어 이후 설치가 깨진다.

### 3.1 작업 폴더 만들기

```bash
mkdir -p ~/benchmark
cd ~/benchmark
```

### 3.2 측정 코드 복사(USB로 윈도우 연결 시)

e-Class에서 받은 `jetson_classification_benchmark.py`를 **Windows PowerShell**에서 Jetson으로 복사한다.

```powershell
scp "C:\Users\user\Downloads\jetson_classification_benchmark.py" hansungai@192.168.55.1:~/benchmark/
```

> [!WARNING]
> Windows `Downloads`에 같은 이름의 파일이 이미 있으면, 새로 받은 파일은 `jetson_classification_benchmark (1).py`로 저장된다. 이 경우 **예전 파일을 지우고 새 파일 이름을 바꾼 뒤** 복사한다. 예전 파일로 실행하면 7절의 `TypeError`가 발생한다.

Jetson에서 **최신 버전인지 확인**한다.

```bash
grep -c "TypeError, UnicodeError" ~/benchmark/jetson_classification_benchmark.py
```

`2`가 나오면 최신 버전이다. `0`이 나오면 예전 파일이므로 다시 복사한다.

### 3.3 가상환경 생성

```bash
cd ~/benchmark
/usr/bin/python3.10 -m venv venv
```

- 프롬프트에 `(base)`가 보이면 conda가 켜진 상태다. `conda deactivate`를 먼저 실행한다.
- `ensurepip is not available` 오류가 나면 교수자에게 문의한다.

### 3.4 PyTorch 설치

```bash
cd ~/benchmark

# (1) pip 업데이트
./venv/bin/python -m pip install --upgrade pip

# (2) Jetson 전용 CUDA torch (반드시 --no-deps 와 --index-url 을 함께 사용)
./venv/bin/python -m pip install --no-deps torch==2.8.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# (3) torch 의존성 (numpy 는 반드시 2 미만)
./venv/bin/python -m pip install filelock "typing-extensions>=4.10" "sympy>=1.13.3" \
    networkx jinja2 fsspec "numpy<2"
```

> [!CAUTION]
> Jetson에서 `pip install torch`를 index 지정 없이 실행하면 **GPU를 쓰지 못하는 PyPI용 torch**가 설치된다. 이 가상환경에는 위 패키지 외에 torchvision, ultralytics 등 **다른 패키지를 추가로 설치하지 않는다.** 이런 패키지가 torch를 PyPI판으로 바꿔 버릴 수 있다.

### 3.5 설치 확인

```bash
./venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

정상이라면 경고 없이 아래처럼 출력된다.

```text
2.8.0 12.6 True Orin
```

| 출력 | 조치 |
|---|---|
| `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` | `./venv/bin/python -m pip install "numpy<2"` 후 다시 확인 |
| 세 번째 값이 `False`, 또는 두 번째 값이 `None` | `./venv/bin/python -m pip uninstall -y torch` 후 4.5의 (2)부터 다시 실행 |
| `libcusparseLt.so.0` / `libopenblas` 관련 오류 | 시스템 라이브러리 문제이므로 교수자에게 문의 |

> [!TIP]
> 이 가이드는 `source venv/bin/activate`를 쓰지 않는다. 대신 **`./venv/bin/python`으로 가상환경의 Python을 직접 지정**한다. activate는 새 터미널이나 SSH 창을 열 때마다 풀리지만, 경로를 직접 쓰면 항상 같은 환경이 사용된다. `./venv/...`는 `~/benchmark` 폴더 안에서만 동작하므로, 다른 폴더에서는 `~/benchmark/venv/bin/python`처럼 전체 경로를 쓴다.

## 4. 측정 전 확인

교수자가 사전에 전력 모드 `MAXN_SUPER`와 최대 클럭 설정을 적용한다. 학생은 원칙적으로 **sudo 설정을 변경하지 않는다.** 측정 중에는 다른 GPU 작업을 종료한다.

```bash
cd ~/benchmark

# 전력 모드 및 클럭 상태 확인
nvpmodel -q
jetson_clocks --show

# 배포 파일 확인
ls -l ~/benchmark/jetson_classification_benchmark.py
```

- `MAXN_SUPER`가 표시되지 않거나 클럭 설정이 다른 것으로 의심되면 **임의 수정하지 말고 문의**한다.
- `jetson_clocks --show` 조회에 권한이 필요한 경우 교수자에게 상태 확인을 요청한다.
- 배포된 `jetson_classification_benchmark.py`는 **수정하지 않는다**.

## 5. 학생 실험 절차 — 정확히 3회 실행

각 조는 아래 `XX`를 **자신의 조 번호(01~14)** 로 변경한다. 예시는 **1조** 기준이다.

```bash
cd ~/benchmark

# 1차
./venv/bin/python jetson_classification_benchmark.py --out team01_run1.json

# 2차
./venv/bin/python jetson_classification_benchmark.py --out team01_run2.json

# 3차
./venv/bin/python jetson_classification_benchmark.py --out team01_run3.json

# 생성 결과 확인
ls -lh team01_run*.json
```

- 각 실행은 **B=1과 B=16을 자동으로 모두 측정**한다.
- 한 번 실행할 때 배치 크기별 100회 Warm-up, GPU 측정 200회 × 3 trial, 호스트 측정 200회를 수행한다.
- **총 3번 실행한 JSON을 모두 제출**한다. 가장 빠른 결과만 골라 제출하지 않는다.
- 세 번 모두 **같은 보드, 같은 가상환경, 같은 전력·클럭 설정**으로 측정한다.
- 측정 도중 GPU를 사용하는 다른 프로그램을 실행하지 않는다.
- 오류가 발생한 실행 결과는 제출하지 말고 오류를 해결한 뒤 해당 회차를 다시 측정한다.

### 정상 출력 예시

```text
B= 1 | median=1.1180 ms/batch | p95=1.1466 ms/batch | throughput=894.42 img/s | wall=1.0993 ms
B=16 | median=5.4358 ms/batch | p95=5.4849 ms/batch | throughput=2943.44 img/s | wall=5.5999 ms
Saved: team01_run1.json
```

위 수치는 실습 중 측정된 한 보드의 **예시**이며, 모든 보드에서 동일하게 나와야 한다는 의미는 아니다.

## 6. e-Class 제출 안내

### 제출 파일

**각 조는 아래 JSON 파일 3개를 하나의 ZIP 파일로 압축해 e-Class 과제 탭에 제출한다.**

```text
team01_benchmark.zip
├── team01_run1.json
├── team01_run2.json
└── team01_run3.json
```

예: 7조는 `team07_benchmark.zip` 안에 `team07_run1.json`, `team07_run2.json`, `team07_run3.json`을 넣는다.

**제출하지 않아도 되는 것:** Python 코드, 가상환경 폴더, 모델 가중치, 이미지 데이터셋, 별도의 보고서, 스크린샷, `compare_teams.csv`.

### Jetson → Windows로 결과 가져오기

**Windows PowerShell**에서 실행한다. Jetson SSH 터미널에서 실행하면 안 된다.

```powershell
scp "hansungai@192.168.55.1:/home/hansungai/benchmark/team01_run*.json" "C:\Users\user\Downloads\"
```

- Windows에 세 개의 JSON이 다운로드되면 파일을 선택하여 **ZIP으로 압축**한다.
- ZIP 파일명을 `team01_benchmark.zip`으로 지정한다.
- 조 대표 1명이 **e-Class → 해당 교과목 → 과제 탭**에 ZIP을 첨부해 제출한다.
- e-Class의 마감 기한을 준수한다.

**주의:** Windows 경로가 포함된 `scp` 명령은 반드시 **Windows PowerShell**에서 실행한다. Jetson Linux 터미널에서는 `$env:USERPROFILE`이나 `C:\Users\...` 경로를 그대로 사용할 수 없다.

### 제출 전 체크리스트

- [ ] 파일명이 `teamXX_run1.json`, `teamXX_run2.json`, `teamXX_run3.json`으로 되어 있다.
- [ ] JSON 파일이 모두 3개이고, 각각 `B=1`과 `B=16` 결과가 들어 있다.
- [ ] 최신 측정 코드를 사용했고, 코드를 변경하지 않았다.
- [ ] 세 번 모두 동일 Jetson / 동일한 가상환경(`~/benchmark/venv`)에서 실행했다.
- [ ] `teamXX_benchmark.zip`을 e-Class **과제 탭**에 조 대표가 제출했다.

## 7. 오류 발생 시

| 증상 | 원인 | 해결 |
|---|---|---|
| `REMOTE HOST IDENTIFICATION HAS CHANGED!` | 다른 보드에 연결했거나 보드가 재설치됨 | Windows에서 `ssh-keygen -R 192.168.55.1` 실행 후 재접속 |
| `SSL: CERTIFICATE_VERIFY_FAILED ... self-signed certificate` | Jetson이 웹 로그인이 필요한 Wi-Fi에 연결됨 | 설치를 중단하고 교수자에게 문의 |
| `No module named 'torch'` | 가상환경이 아닌 Python으로 실행함 (`python`, `sudo python` 등) | `cd ~/benchmark` 후 `./venv/bin/python ...`으로 실행 |
| `can't open file ...benchmark.py` | 다른 폴더에서 실행했거나 파일이 없음 | `cd ~/benchmark` 후 `ls -l jetson_classification_benchmark.py`로 확인 |
| `CUDA unavailable` | GPU를 쓰지 못하는 torch가 설치됨 | 4.6의 표를 따라 재설치. 그래도 안 되면 측정하지 말고 문의 |
| `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` | NumPy 2가 설치됨 | `./venv/bin/python -m pip install "numpy<2"` |
| `TypeError: can't concat NoneType to bytes` (온도 센서 읽기) | **예전 버전 측정 코드**를 사용함 | 4.3을 따라 최신 파일로 교체. 최신 코드는 센서 읽기 오류를 건너뛴다 |

---

## 8. 교수자용

### 8.1 보드 사전 준비 (sudo, 보드당 1회)

```bash
# 시간 동기화 (시계가 틀리면 인증서 오류가 날 수 있음)
sudo timedatectl set-ntp true

# 가상환경 생성에 필요한 패키지
sudo apt-get update
sudo apt-get install -y python3.10-venv

# 전력 모드: MAXN_SUPER 의 모드 ID 확인 후 설정 (보드에서 ID 2인 경우)
sudo nvpmodel -q --verbose | grep -i maxn
sudo nvpmodel -m 2
sudo jetson_clocks          # 재부팅을 요구하면 재부팅 후 다시 실행
nvpmodel -q                 # MAXN_SUPER 확인
```

- **네트워크:** 교내 무선망을 쓸 경우 `HANSUNG-INFO`(웹 로그인 방식)가 아니라 802.1X 방식의 `hansung` 또는 `eduroam` 프로필로 연결되어 있는지 확인한다. `HANSUNG-INFO`에 붙어 있으면 pip 설치가 인증서 오류로 실패한다. 이 경우 `HANSUNG-INFO`의 자동 연결을 꺼 둔다. 유선 LAN을 쓸 수 있으면 유선이 가장 안정적이다.
- `libcusparseLt.so.0` 오류가 나는 보드에는 NVIDIA CUDA 저장소를 등록한 뒤 `sudo apt-get install -y cusparselt-cuda-12`를 설치한다. 시범 보드에서는 torch 2.8.0을 쓸 때 필요하지 않았다.

### 8.2 결과 집계

학생의 ZIP을 압축 해제하여 14개 조의 JSON **총 42개**를 `results` 폴더에 모은다.

```text
jetson_benchmark/
├── compare_teams.py
└── results/
    ├── team01_run1.json
    ├── team01_run2.json
    ├── team01_run3.json
    ├── ...
    └── team14_run3.json
```

Windows PowerShell 또는 Linux에서:

```bash
python compare_teams.py results/*.json
```

제공된 `compare_teams.py`는 **동일 조의 3회 측정값을 먼저 중앙값으로 합친 다음**, 조별 대표값 14개를 비교한다. 결과는 `compare_teams.csv`로 저장된다. Python 표준 라이브러리만 사용하므로 교수자 PC에서 PyTorch를 설치할 필요가 없다.

| 대표 지표 | 집계 방식 |
|---|---|
| B=1 GPU 추론시간 | 3개 JSON의 `median_ms` 중앙값 |
| B=16 처리량 | 3개 JSON의 `images_per_second` 중앙값 |
| 보드 간 편차 | 조별 대표값의 전체 중앙값 대비 편차(%) |
| 측정 안정성 | 3회 결과의 최댓값-최솟값 범위(중앙값 대비 %) |

**점검 기준(수업 운영용, 공식 보증 기준 아님)**

- 편차 **±5% 이내**: 유사(`OK`)
- 편차 **5~10%**: 재측정
- 편차 **10% 초과**: 전력 모드·클럭·발열·백그라운드 프로그램 및 버전 점검

집계 전에 각 JSON의 `torch_version`이 모두 `2.8.0`인지 확인한다. 버전이 다른 조는 재측정 대상이다.

B=1에서는 CPU/Python/커널 실행 오버헤드의 영향이 상대적으로 크고, B=16에서는 GPU 계산·메모리 측면의 차이가 더 드러날 수 있다. GPU 추론 이외의 통신·다운로드·데이터 로딩 속도는 본 실험에서 비교하지 않는다.

> JSON에 기록된 `sysfs_before`와 `sysfs_after`는 **측정 전·종료 시점의 센서 스냅샷**이다. 실험 도중의 실제 최고 온도를 기록한 값은 아니다. 센서 조회가 실패하면 해당 값은 누락될 수 있으며, 추론 시간 결과와 구분해 해석한다.

---

**핵심 순서: 접속 → `~/benchmark` 폴더 → 최신 코드 복사 → venv 생성 → torch 2.8.0 + `numpy<2` 설치 → 3회 실행 → JSON 3개 → ZIP 1개 → e-Class 과제 탭 제출.**
