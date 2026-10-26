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
| 보드 | Jetson Orin Nano 8GB Super |
| 전력 모드 | `MAXN_SUPER` |
| 클럭 | `jetson_clocks` 적용 |
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

### 측정 지표

- `median_ms`: CUDA Event 기준 **GPU 스트림 경과시간**(ms/batch). 낮을수록 빠르다.
- `p95_ms`: GPU 스트림 경과시간의 95백분위수.
- `wall_median_ms`: CPU/Python 호출과 GPU 동기화 시간을 포함한 호스트 측 경과시간(ms/batch).
- `images_per_second`: `batch_size × 1000 / median_ms`로 계산한 **GPU 시간 기준 처리량**. 높을수록 빠르다.

**비교의 중심 지표는 B=1의 `median_ms`와 B=16의 `images_per_second`이다.**

## 3. 측정 전 확인

교수자가 사전에 전력 모드 `MAXN_SUPER`와 최대 클럭 설정을 적용한다. 학생은 원칙적으로 **sudo 설정을 변경하지 않는다.** 측정 중에는 다른 GPU 작업을 종료한다.

Jetson 터미널에서:

```bash
# 벤치마크 환경 활성화
source ~/benchmark_env/bin/activate
cd ~

# CUDA 지원 PyTorch 확인 (마지막 값이 True여야 함)
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# 전력 모드 및 클럭 상태 확인
nvpmodel -q
jetson_clocks --show

# 배포 파일 확인
ls -l ~/jetson_classification_benchmark.py
```

- `torch.cuda.is_available()`이 `False`이거나 `ModuleNotFoundError`가 발생하면 **측정하지 말고 교수자에게 문의**한다.
- `MAXN_SUPER`가 표시되지 않거나 클럭 설정이 다른 것으로 의심되면 **임의 수정하지 말고 문의**한다.
- `jetson_clocks --show` 조회에 권한이 필요한 경우 교수자에게 상태 확인을 요청한다.
- Python 실행 파일은 배포된 `jetson_classification_benchmark.py`를 **수정하지 않는다**.

> **교수자 초기 설정 참고:** `sudo nvpmodel -q`로 모드 ID를 확인한 뒤, 해당 보드에서 `MAXN_SUPER`가 ID 2인 경우 `sudo nvpmodel -m 2`를 설정한다. 필요하면 재부팅 후 `sudo jetson_clocks`를 실행한다. 모든 조에 동일하게 적용한다.

## 4. 학생 실험 절차 — 정확히 3회 실행

각 조는 아래 `XX`를 **자신의 조 번호(01~14)** 로 변경한다. 예시는 **1조** 기준이다.

```bash
source ~/benchmark_env/bin/activate
cd ~

# 1차
python jetson_classification_benchmark.py --out team01_run1.json

# 2차
python jetson_classification_benchmark.py --out team01_run2.json

# 3차
python jetson_classification_benchmark.py --out team01_run3.json

# 생성 결과 확인
ls -lh team01_run*.json
```

- 각 실행은 **B=1과 B=16을 자동으로 모두 측정**한다.
- 한 번 실행할 때 배치 크기별 100회 Warm-up, GPU 측정 200회 × 3 trial, 호스트 측정 200회를 수행한다.
- **총 3번 실행한 JSON을 모두 제출**한다. 가장 빠른 결과만 골라 제출하지 않는다.
- 세 번 모두 **같은 보드, 같은 Python 환경, 같은 전력·클럭 설정**으로 측정한다.
- 측정 도중 GPU를 사용하는 다른 프로그램을 실행하지 않는다.
- 오류가 발생한 실행 결과는 제출하지 말고 오류를 해결한 뒤 해당 회차를 다시 측정한다.

### 정상 출력 예시

```text
B= 1 | median=1.1180 ms/batch | p95=1.1466 ms/batch | throughput=894.42 img/s | wall=1.0993 ms
B=16 | median=5.4358 ms/batch | p95=5.4849 ms/batch | throughput=2943.44 img/s | wall=5.5999 ms
Saved: team01_run1.json
```

위 수치는 실습 중 측정된 한 보드의 **예시**이며, 모든 보드에서 동일하게 나와야 한다는 의미는 아니다.

## 5. e-Class 제출 안내

### 제출 파일

**각 조는 아래 JSON 파일 3개를 하나의 ZIP 파일로 압축해 e-Class 과제 탭에 제출한다.**

```text
team01_benchmark.zip
├── team01_run1.json
├── team01_run2.json
└── team01_run3.json
```

예: 7조는 `team07_benchmark.zip` 안에 `team07_run1.json`, `team07_run2.json`, `team07_run3.json`을 넣는다.

**제출하지 않아도 되는 것:** Python 코드, 모델 가중치, 이미지 데이터셋, 별도의 보고서, 스크린샷, `compare_teams.csv`.

### Jetson → Windows로 결과 가져오기

**Windows PowerShell**에서 실행한다. (Jetson SSH 터미널이 아님.)

```powershell
scp "hansungai@192.168.55.1:/home/hansungai/team01_run*.json" "C:\Users\user\Downloads\"
```

- Windows에 세 개의 JSON이 다운로드되면 파일을 선택하여 **ZIP으로 압축**한다.
- ZIP 파일명을 `team01_benchmark.zip`으로 지정한다.
- 조 대표 1명이 **e-Class → 해당 교과목 → 과제 탭**에 ZIP을 첨부해 제출한다.
- e-Class의 마감 기한을 준수한다.

**주의:** Windows 경로가 포함된 `scp` 명령은 반드시 **Windows PowerShell**에서 실행한다. Jetson Linux 터미널에서는 `$env:USERPROFILE`이나 `C:\Users\...` 경로를 그대로 사용할 수 없다.

### 제출 전 체크리스트

- [ ] 파일명이 `teamXX_run1.json`, `teamXX_run2.json`, `teamXX_run3.json`으로 되어 있다.
- [ ] JSON 파일이 모두 3개이고, 각각 `B=1`과 `B=16` 결과가 들어 있다.
- [ ] 측정 코드를 변경하지 않았다.
- [ ] 세 번 모두 동일 Jetson / 동일한 환경에서 실행했다.
- [ ] `teamXX_benchmark.zip`을 e-Class **과제 탭**에 조 대표가 제출했다.

## 6. 교수자용 결과 집계

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

B=1에서는 CPU/Python/커널 실행 오버헤드의 영향이 상대적으로 크고, B=16에서는 GPU 계산·메모리 측면의 차이가 더 드러날 수 있다. GPU 추론 이외의 통신·다운로드·데이터 로딩 속도는 본 실험에서 비교하지 않는다.

> JSON에 기록된 `sysfs_before`와 `sysfs_after`는 **측정 전·종료 시점의 센서 스냅샷**이다. 실험 도중의 실제 최고 온도를 기록한 값은 아니다. 센서 조회가 실패하면 해당 값은 누락될 수 있으며, 추론 시간 결과와 구분해 해석한다.

## 7. 오류 발생 시

- **`No module named 'torch'`**: `source ~/benchmark_env/bin/activate`를 실행했는지 확인하고, 설치 문제가 있으면 교수자에게 문의한다.
- **`can't open file ...benchmark.py`**: `ls ~/jetson_classification_benchmark.py`로 파일이 Jetson에 있는지 확인한다.
- **`CUDA unavailable`**: CUDA 지원 Jetson용 PyTorch가 준비되지 않았으므로 측정하지 않는다.
- **`TypeError: can't concat NoneType to bytes` (온도 센서 읽기)**: 예전 버전 코드에서 발생하는 문제다. 최신 배포본 스크립트로 교체한다. 제공된 최신 스크립트는 센서 읽기 오류를 건너뛰도록 수정되었다.

**핵심 제출사항: 조별 3회 실행 → JSON 3개 → ZIP 1개 → e-Class 과제 탭 제출.**
