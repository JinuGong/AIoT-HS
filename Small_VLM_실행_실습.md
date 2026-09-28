# Jetson Small VLM 실행 실습

## 1. 실습 목표

이 실습에서는 Jetson Orin Nano에서 `llama.cpp`를 CUDA backend로 직접 빌드하고, GGUF 형식의 경량 Vision-Language Model(VLM)을 실행합니다.

실습 모델은 다음 조합을 사용합니다.

```text
Language model : SmolVLM2-2.2B-Instruct Q4_K_M
Vision model   : SmolVLM2-2.2B-Instruct mmproj Q8_0
Runtime        : llama.cpp / llama-mtmd-cli
Device         : Jetson Orin Nano GPU
```

전체 실습 흐름은 다음과 같습니다.

```text
새 VLM 작업 환경 생성
    ↓
llama.cpp CUDA 빌드
    ↓
GGUF language model 다운로드
    ↓
mmproj vision projector 다운로드
    ↓
학생이 준비한 이미지로 직접 실행
    ↓
한 질문의 cold-start 비용 측정
    ↓
여러 질문 반복 실험
    ↓
응답·실행 시간·메모리·전력 분석
```


## 2. VLM 실행 구조

`llama.cpp`의 멀티모달 실행에는 일반적으로 서로 대응하는 두 GGUF 파일이 필요합니다.

```text
입력 이미지
    ↓
mmproj GGUF
Vision encoding + projection
    ↓
Image embeddings
    ↓
Language model GGUF
Prompt evaluation + text generation
    ↓
텍스트 응답
```

| 구성 요소 | 역할 |
|---|---|
| Language model GGUF | 텍스트 이해 및 응답 생성 |
| `mmproj` GGUF | Vision encoder 및 multimodal projection에 필요한 가중치를 포함하며, 이미지를 language model이 사용할 수 있는 embedding으로 변환 |
| `llama-mtmd-cli` | 이미지, 질문, 모델을 연결하는 멀티모달 CLI |

두 GGUF는 동일한 모델 계열에 맞는 조합을 사용해야 합니다.

## 3. 새 작업 디렉터리와 가상환경 생성

VLM 실습용 디렉터리를 새로 생성합니다.

```bash
cd ~
mkdir -p ~/vlm_lab
cd ~/vlm_lab
```

Python 가상환경을 생성하고 활성화합니다.

```bash
python3 -m venv vlm_env
source vlm_env/bin/activate
```

Python 버전을 확인합니다.

```bash
python --version
```

Jetson Ubuntu 22.04 환경에서는 일반적으로 Python 3.10 계열이 표시됩니다.

이 가상환경은 VLM 추론 엔진을 실행하기 위한 환경이 아닙니다. `llama.cpp`는 C++로 빌드됩니다. Python 환경은 Hugging Face 모델 다운로드와 이후 로그 분석에 사용합니다.

## 4. Python 유틸리티 설치

```bash
python -m pip install --upgrade pip
pip install huggingface_hub pandas numpy
```

Hugging Face CLI가 설치됐는지 확인합니다.

```bash
hf --help
```

이번 실습에서는 다음 패키지가 필요하지 않습니다.

- PyTorch
- torchvision
- Transformers
- Jetson용 PyTorch wheel

VLM 추론은 GGUF 모델과 CUDA로 빌드한 `llama.cpp`에서 수행합니다.

## 5. 빌드 도구 설치 및 확인

필요한 패키지를 설치합니다.

```bash
sudo apt update
sudo apt install -y git cmake build-essential time
```

각 도구가 정상적으로 실행되는지 확인합니다.

```bash
git --version
cmake --version
g++ --version
nvcc --version
```

GNU `time`의 실제 경로도 확인합니다.

```bash
which time
ls -l /usr/bin/time
```

`run_vlm.sh`에서는 shell 내장 `time`이 아니라 `/usr/bin/time -v`를 사용하여 전체 process 시간과 최대 메모리 사용량을 기록합니다.

## 6. `llama.cpp` 다운로드

```bash
cd ~/vlm_lab
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

다운로드한 소스의 commit을 확인합니다.

```bash
git rev-parse HEAD
```

`llama.cpp`의 멀티모달 기능은 빠르게 변경될 수 있습니다. 실험 재현을 위해 실제 사용한 commit hash를 결과와 함께 보존합니다.

## 7. Jetson CUDA 빌드

> [!IMPORTANT]
> **수업 전 사전 준비 권장**
>
> `llama.cpp`의 CUDA backend 첫 빌드는 Jetson Orin Nano에서 상당히 오래 걸릴 수 있습니다.
> 특히 attention, quantization, matrix multiplication 관련 CUDA kernel들을 다수 컴파일하므로,
> 수업 시간에 처음부터 빌드하면 실습 시간이 크게 소모될 수 있습니다.
>
> 따라서 **교수자 또는 학생은 수업 전에 아래 CUDA 빌드 과정을 미리 완료**해 두는 것을 권장합니다.
>
> 사전 준비 완료 여부는 다음 명령으로 확인합니다.
>
> ```bash
> cd ~/vlm_lab/llama.cpp
> ls -lh build/bin/llama-mtmd-cli
> ./build/bin/llama-mtmd-cli --version
> ```
>
> `build/bin/llama-mtmd-cli`가 존재하고 버전 정보가 정상적으로 출력되면 수업 중에는 다시 전체 빌드를 수행할 필요가 없습니다.
>
> 참고로 오래 걸리는 부분은 `git clone` 자체가 아니라 다음 **CUDA 컴파일 단계**입니다.
>
> ```bash
> cmake --build build --target llama-mtmd-cli -j4
> ```
>
> 빌드 중 다음과 같이 CUDA object가 계속 생성되고 있다면 정상적으로 진행 중입니다.
>
> ```text
> [26%] Building CUDA object ...
> [28%] Building CUDA object ...
> [29%] Building CUDA object ...
> ```
>
> 빌드가 메모리 부족으로 중단되면 `-j4` 대신 `-j2`로 다시 실행합니다. 이미 완료된 object는 재사용됩니다.


Jetson Orin의 CUDA compute capability에 맞춰 CUDA backend를 활성화합니다.

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release
```

`llama-mtmd-cli` target을 빌드합니다.

```bash
cmake --build build --target llama-mtmd-cli -j4
```

> [!NOTE]
> 8GB Jetson에서 모든 CPU core를 사용해 병렬 빌드하면 메모리가 부족해질 수 있습니다. 이 문서에서는 비교적 안전한 `-j4`를 사용합니다.

빌드 결과를 확인합니다.

```bash
ls -lh build/bin/llama-mtmd-cli
./build/bin/llama-mtmd-cli --version
```

멀티모달 옵션이 포함되어 있는지도 확인합니다.

```bash
./build/bin/llama-mtmd-cli --help | grep -E -- "--mmproj|--image"
```

다음 두 옵션이 모두 표시되어야 합니다.

```text
--mmproj
--image
```

## 8. 모델 디렉터리 생성

```bash
mkdir -p ~/vlm_lab/models/smolvlm2
cd ~/vlm_lab/models/smolvlm2
```

## 9. SmolVLM2 모델 다운로드

Language model Q4_K_M 파일을 다운로드합니다.

```bash
hf download \
  ggml-org/SmolVLM2-2.2B-Instruct-GGUF \
  SmolVLM2-2.2B-Instruct-Q4_K_M.gguf \
  --local-dir .
```

대응하는 Q8_0 vision projector를 다운로드합니다.

```bash
hf download \
  ggml-org/SmolVLM2-2.2B-Instruct-GGUF \
  mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf \
  --local-dir .
```

다운로드한 파일을 확인합니다.

```bash
ls -lh *.gguf
```

다음 두 파일이 있어야 합니다.

```text
SmolVLM2-2.2B-Instruct-Q4_K_M.gguf
mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf
```

다운로드가 중단되면 같은 `hf download` 명령을 다시 실행합니다. 이미 받은 데이터는 가능한 범위에서 재사용됩니다.

## 10. 실험 이미지 준비

학생이 직접 촬영하거나 보유한 JPG 또는 PNG 이미지 한 장을 준비합니다. 개인정보, 얼굴, 차량 번호, 문서의 민감 정보가 포함된 이미지는 사용하지 않습니다.

이미지를 다음 경로에 저장합니다.

```text
~/vlm_lab/images/test.jpg
```

디렉터리를 먼저 만듭니다.

```bash
mkdir -p ~/vlm_lab/images
```

Windows PC에서 VS Code의 Jetson 원격 탐색기를 사용하거나 `scp`로 파일을 전송할 수 있습니다. 전송 후 확인합니다.

```bash
file ~/vlm_lab/images/test.jpg
ls -lh ~/vlm_lab/images/test.jpg
```

`file` 명령 결과에 JPEG 또는 PNG 이미지로 표시되어야 합니다. PNG를 사용하는 경우 이후 `IMAGE` 변수의 확장자도 실제 파일에 맞게 변경합니다.

## 11. 실습 디렉터리 구조

```text
~/vlm_lab/
├─ vlm_env/
├─ llama.cpp/
│  └─ build/bin/llama-mtmd-cli
├─ models/
│  └─ smolvlm2/
│     ├─ SmolVLM2-2.2B-Instruct-Q4_K_M.gguf
│     └─ mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf
├─ images/
│  └─ test.jpg
├─ scripts/
│  ├─ run_vlm.sh
│  ├─ run_vlm_questions.sh
│  └─ summarize_vlm.py
├─ questions.txt
└─ results/
```

## 12. 단일 질문 실행 스크립트 생성

스크립트 디렉터리를 생성합니다.

```bash
mkdir -p ~/vlm_lab/scripts
```

`run_vlm.sh`를 생성합니다.

```bash
cat > ~/vlm_lab/scripts/run_vlm.sh <<'SH'
#!/usr/bin/env bash

# One cold process per question:
# model load + mmproj load + image encode + prompt eval + generation.

set -euo pipefail
export LC_ALL=C

MODEL=${1:?Usage: bash run_vlm.sh model.gguf mmproj.gguf image.jpg [prompt]}
MMPROJ=${2:?Matching mmproj.gguf required}
IMAGE=${3:?Image path required}
PROMPT=${4:-Describe this image in one sentence.}

NGL=${NGL:-5}
CTX=${CTX:-512}
NPRED=${NPRED:-32}
MMPROJ_OFFLOAD=${MMPROJ_OFFLOAD:-1}
BIN=${VLM_BIN:-./build/bin/llama-mtmd-cli}
OUT=${OUT_DIR:-vlm_$(date +%Y%m%d_%H%M%S)}

for path in "$MODEL" "$MMPROJ" "$IMAGE"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing file: $path" >&2
        exit 1
    fi
done

if [[ ! -x "$BIN" ]]; then
    echo "Binary not found: $BIN" >&2
    echo "Run from the llama.cpp root or set VLM_BIN." >&2
    exit 1
fi

if [[ ! -x /usr/bin/time ]]; then
    echo "GNU time is missing: sudo apt install time" >&2
    exit 1
fi

HELP=$($BIN --help 2>&1 || true)

if ! grep -q -- "--mmproj" <<< "$HELP" || ! grep -q -- "--image" <<< "$HELP"; then
    echo "$BIN does not support --mmproj and --image." >&2
    exit 1
fi

mkdir -p "$OUT"

$BIN --version > "$OUT/version.txt" 2>&1 || true
git rev-parse HEAD > "$OUT/llama_commit.txt" 2>/dev/null || echo unknown > "$OUT/llama_commit.txt"
sha256sum "$MODEL" "$MMPROJ" "$IMAGE" > "$OUT/inputs.sha256"
printf '%s\n' "$PROMPT" > "$OUT/prompt.txt"
printf '{"ngl":%s,"ctx":%s,"n_predict":%s,"mmproj_offload":%s}\n' \
    "$NGL" "$CTX" "$NPRED" "$MMPROJ_OFFLOAD" > "$OUT/conditions.json"

MM_ARGS=()
if [[ "$MMPROJ_OFFLOAD" == "0" ]]; then
    MM_ARGS+=(--no-mmproj-offload)
fi

/usr/bin/time -v -o "$OUT/process_time.txt" \
    "$BIN" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --image "$IMAGE" \
    -p "$PROMPT" \
    -n "$NPRED" \
    -c "$CTX" \
    --temp 0 \
    --seed 0 \
    -ngl "$NGL" \
    "${MM_ARGS[@]}" \
    < /dev/null \
    > "$OUT/response.txt" \
    2> "$OUT/runtime.log"

if [[ ! -s "$OUT/response.txt" ]]; then
    echo "INVALID RUN: empty response. Check $OUT/runtime.log." >&2
    exit 2
fi

printf 'Saved process-level logs to %s\n' "$OUT"
SH
```

실행 권한을 부여합니다.

```bash
chmod +x ~/vlm_lab/scripts/run_vlm.sh
```

### 저장되는 결과

| 파일 | 내용 |
|---|---|
| `response.txt` | VLM이 생성한 응답 |
| `runtime.log` | 모델 로딩, image encode, prompt eval, decode 로그 |
| `process_time.txt` | 전체 process 시간, CPU 사용률, 최대 RSS |
| `version.txt` | `llama-mtmd-cli` 버전 |
| `llama_commit.txt` | 빌드에 사용한 `llama.cpp` commit |
| `inputs.sha256` | 모델, projector 및 이미지의 SHA-256 |
| `prompt.txt` | 입력 질문 |
| `conditions.json` | GPU layer, context, 생성 길이 조건 |

## 13. 여러 질문 실행 스크립트 생성

`run_vlm_questions.sh`를 생성합니다.

```bash
cat > ~/vlm_lab/scripts/run_vlm_questions.sh <<'SH'
#!/usr/bin/env bash

# Run run_vlm.sh once per non-empty line of the questions file.

set -euo pipefail

MODEL=${1:?Usage: bash run_vlm_questions.sh model.gguf mmproj.gguf image.jpg questions.txt}
MMPROJ=${2:?Matching mmproj.gguf required}
IMAGE=${3:?Image path required}
QUESTIONS=${4:?Questions file required}

OUT_ROOT=${OUT_ROOT:-vlm_$(date +%Y%m%d_%H%M%S)}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -f "$QUESTIONS" ]]; then
    echo "Questions file not found: $QUESTIONS" >&2
    exit 1
fi

index=0

while IFS= read -r question || [[ -n "$question" ]]; do
    [[ -z "$question" ]] && continue

    index=$((index + 1))

    OUT_DIR="$OUT_ROOT/q$index" \
        bash "$HERE/run_vlm.sh" \
        "$MODEL" "$MMPROJ" "$IMAGE" "$question" \
        || echo "q$index failed. Check $OUT_ROOT/q$index." >&2
done < "$QUESTIONS"

echo "Completed $index question(s)."
echo "Result root: $OUT_ROOT"
echo "Summarize with: python $HERE/summarize_vlm.py $OUT_ROOT"
SH
```

실행 권한을 부여합니다.

```bash
chmod +x ~/vlm_lab/scripts/run_vlm_questions.sh
```

각 질문은 별도의 process에서 실행됩니다.

```text
Question 1 → 모델 새로 로드 → 추론 → process 종료
Question 2 → 모델 새로 로드 → 추론 → process 종료
Question 3 → 모델 새로 로드 → 추론 → process 종료
```

따라서 이 실험은 이미 모델이 로드된 서버의 steady-state latency가 아니라 모델 로드까지 포함한 **cold VLM execution cost**를 측정합니다.

## 14. 결과 요약 스크립트 생성

`summarize_vlm.py`를 생성합니다.

```bash
cat > ~/vlm_lab/scripts/summarize_vlm.py <<'PY'
import argparse
import csv
import re
from pathlib import Path


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def find_value(text, label):
    pattern = rf"^{re.escape(label)}:\s*(.+)$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else "N/A"


parser = argparse.ArgumentParser()
parser.add_argument("result_root", type=Path)
args = parser.parse_args()

result_dirs = sorted(
    [path for path in args.result_root.glob("q*") if path.is_dir()],
    key=lambda path: int(path.name[1:]) if path.name[1:].isdigit() else path.name,
)

rows = []

for result_dir in result_dirs:
    process_time = read_text(result_dir / "process_time.txt")
    prompt = read_text(result_dir / "prompt.txt").strip()
    response = read_text(result_dir / "response.txt").strip()

    rows.append({
        "run": result_dir.name,
        "prompt": prompt,
        "elapsed": find_value(process_time, "Elapsed (wall clock) time (h:mm:ss or m:ss)"),
        "max_rss_kb": find_value(process_time, "Maximum resident set size (kbytes)"),
        "cpu_percent": find_value(process_time, "Percent of CPU this job got"),
        "response": response.replace("\n", " "),
        "runtime_log": str(result_dir / "runtime.log"),
    })

output_path = args.result_root / "summary.csv"

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "run",
            "prompt",
            "elapsed",
            "max_rss_kb",
            "cpu_percent",
            "response",
            "runtime_log",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Runs  : {len(rows)}")
print(f"Saved : {output_path}")
PY
```

## 15. 질문 파일 생성

```bash
cat > ~/vlm_lab/questions.txt <<'EOF'
Describe this image in one sentence.
What objects can you see in this image?
What is the main subject of this image?
What colors are prominent in this image?
Is there any text visible in the image?
EOF
```

파일을 확인합니다.

```bash
cat ~/vlm_lab/questions.txt
```

빈 줄은 무시되며, 빈 줄이 아닌 한 줄이 질문 하나가 됩니다.

## 16. 경로 변수 설정

`llama.cpp` 디렉터리로 이동합니다.

```bash
cd ~/vlm_lab/llama.cpp
```

모델, projector 및 이미지 경로를 변수에 저장합니다.

```bash
MODEL=~/vlm_lab/models/smolvlm2/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf
MMPROJ=~/vlm_lab/models/smolvlm2/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf
IMAGE=~/vlm_lab/images/test.jpg
```

세 파일이 모두 있는지 확인합니다.

```bash
ls -lh "$MODEL" "$MMPROJ" "$IMAGE"
```

터미널을 새로 열면 shell 변수가 사라지므로 위 세 줄을 다시 실행해야 합니다.

## 17. 스크립트 없이 첫 VLM 실행

Benchmark 스크립트보다 먼저 CLI가 정상 작동하는지 직접 확인합니다.

```bash
./build/bin/llama-mtmd-cli \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --image "$IMAGE" \
  -p "Describe this image in one sentence." \
  -n 32 \
  -c 512 \
  --temp 0 \
  -ngl 5
```

다음을 순서대로 확인합니다.

- Language model이 로드되는가?
- `mmproj`가 로드되는가?
- 이미지가 인코딩되는가?
- 텍스트 응답이 생성되는가?
- CUDA 또는 GPU offload 관련 오류가 없는가?
- Out-of-memory 오류가 없는가?

## 18. GPU 사용 확인과 offload 이해

첫 번째 터미널에서 VLM을 실행하고, 두 번째 터미널에서 다음 명령을 실행합니다.

```bash
sudo tegrastats --interval 100
```

다음을 관찰합니다.

- `GR3D_FREQ`: GPU 사용률
- RAM: CPU와 GPU가 공유하는 Jetson 통합 메모리 사용량
- `VDD_IN`: 보드 입력 전력
- CPU 사용률
- GPU 및 SoC 온도

### `-ngl`의 의미

`-ngl`은 language model의 Transformer layer 중 몇 개를 CUDA backend로 offload할지 지정합니다.

```text
-ngl 0   → language model을 거의 CPU에서 실행
-ngl 5   → 일부 Transformer layer만 GPU로 offload
-ngl 증가 → GPU workload 증가 가능, 메모리 사용량도 증가
```

Jetson Orin Nano는 CPU와 GPU가 물리적으로 같은 통합 메모리를 사용하지만, `llama.cpp`에서는 CPU backend와 CUDA backend에 어떤 tensor를 배치할지 별도로 결정합니다.

### `mmproj` offload의 의미

`mmproj`는 단순히 마지막 projection matrix 하나만 의미하는 것이 아니라, VLM에서 vision encoding 및 multimodal projection에 필요한 구성 요소를 별도 GGUF로 제공하는 파일입니다.

기본 실행에서는 가능한 경우 이 multimodal 경로도 GPU로 offload됩니다.

```bash
-ngl 5
```

반대로 다음 옵션을 추가하면 multimodal 쪽을 CPU에 둡니다.

```bash
--no-mmproj-offload
```

따라서 다음 두 조건을 비교할 수 있습니다.

```text
A. -ngl 5 --no-mmproj-offload
   → language model 일부만 GPU
   → vision/mmproj는 CPU

B. -ngl 5
   → language model 일부 GPU
   → vision/mmproj도 GPU offload 허용
```

### 실제 Jetson Orin Nano 관찰 예

본 실습 환경에서 동일한 SmolVLM2 모델과 동일 이미지로 측정했을 때 다음과 같은 차이가 관찰되었습니다.

```text
mmproj CPU 실행:
image encoding chunk 약 6.1초

mmproj GPU offload:
warm-up 이후 image encoding chunk 약 0.5초
```

첫 GPU 실행에서는 CUDA 초기화, kernel 준비, 메모리 allocation 등의 비용 때문에 첫 chunk가 비정상적으로 오래 걸릴 수 있습니다. 따라서 첫 실행은 warm-up으로 보고, 동일 조건을 다시 실행한 결과를 함께 확인합니다.

`-ngl` 값만으로 GPU 사용 여부를 단정하지 않습니다. `tegrastats`에서 `GR3D_FREQ`가 실제로 상승하는지 확인합니다.

## 19. 단일 질문 cold-process 측정

직접 실행이 성공한 후 측정 스크립트를 실행합니다.

```bash
cd ~/vlm_lab/llama.cpp

OUT_DIR=~/vlm_lab/results/single \
bash ~/vlm_lab/scripts/run_vlm.sh \
  "$MODEL" \
  "$MMPROJ" \
  "$IMAGE" \
  "Describe this image in one sentence."
```

결과를 확인합니다.

```bash
cat ~/vlm_lab/results/single/response.txt
cat ~/vlm_lab/results/single/process_time.txt
```

Runtime log의 주요 timing 관련 줄을 확인합니다.

```bash
grep -Ei "load|image|encode|prompt|eval|sample|total|token" \
  ~/vlm_lab/results/single/runtime.log
```

## 20. 여러 질문 cold-process 측정

```bash
cd ~/vlm_lab/llama.cpp

OUT_ROOT=~/vlm_lab/results/questions \
bash ~/vlm_lab/scripts/run_vlm_questions.sh \
  "$MODEL" \
  "$MMPROJ" \
  "$IMAGE" \
  ~/vlm_lab/questions.txt
```

각 질문의 결과는 다음 구조로 저장됩니다.

```text
~/vlm_lab/results/questions/
├─ q1/
│  ├─ response.txt
│  ├─ runtime.log
│  ├─ process_time.txt
│  └─ ...
├─ q2/
├─ q3/
├─ q4/
└─ q5/
```

질문과 응답을 차례로 확인합니다.

```bash
for directory in ~/vlm_lab/results/questions/q*; do
  echo "===== $directory ====="
  cat "$directory/prompt.txt"
  cat "$directory/response.txt"
done
```

요약 CSV를 생성합니다.

```bash
python ~/vlm_lab/scripts/summarize_vlm.py \
  ~/vlm_lab/results/questions
```

결과를 확인합니다.

```bash
column -s, -t ~/vlm_lab/results/questions/summary.csv | less -S
```

`less`를 종료하려면 `q`를 누릅니다.

## 21. 측정 범위 이해

이 실습에서 질문 하나의 전체 실행 시간은 대략 다음 요소를 모두 포함합니다.

$$
T_{total}
=T_{process\ startup}
+T_{model\ load}
+T_{mmproj\ load}
+T_{vision}
+T_{prompt}
+T_{decode}
$$

| 구간 | 의미 |
|---|---|
| Process startup | 새 process 생성 및 초기화 |
| Model load | Language model을 메모리에 로드 |
| mmproj load | Vision projector 로드 |
| Vision encode | 이미지를 embedding으로 변환 |
| Prompt evaluation | 이미지 embedding과 질문 처리 |
| Decode | 출력 token 생성 |

`process_time.txt`는 위 과정 전체의 cold-process 비용을 기록합니다. `runtime.log`는 현재 빌드가 출력하는 세부 timing을 제공합니다. `llama.cpp` 로그 형식은 commit에 따라 달라질 수 있으므로 특정 세부 지표가 출력되지 않으면 `N/A`로 기록합니다.

## 22. 기록할 실험 지표

| 항목 | 기록 내용 |
|---|---|
| Model | SmolVLM2-2.2B-Instruct |
| Language quantization | Q4_K_M |
| mmproj precision | Q8_0 |
| mmproj offload | GPU / CPU (`--no-mmproj-offload`) |
| Model file size | `ls -lh` 결과 |
| mmproj file size | `ls -lh` 결과 |
| `llama.cpp` commit | `llama_commit.txt` |
| Whole process time | `process_time.txt` |
| Maximum RSS | `process_time.txt` |
| Image encode time | `runtime.log`에 출력되는 경우 |
| Prompt evaluation | `runtime.log`에 출력되는 경우 |
| Decode speed | token/s 로그가 출력되는 경우 |
| Response | `response.txt` |
| GPU/RAM/Power | `tegrastats` 관찰값 |

## 23. 결과 기록표

| Question | 전체 시간 | 최대 RSS | Image encode | Prompt eval | Decode tok/s | 응답 품질 |
|---|---:|---:|---:|---:|---:|---|
| 이미지 한 문장 설명 |  |  |  |  |  |  |
| 객체 나열 |  |  |  |  |  |  |
| 주요 대상 |  |  |  |  |  |  |
| 주요 색상 |  |  |  |  |  |  |
| 이미지 속 문자 |  |  |  |  |  |  |

추가 실행 조건도 기록합니다.

```text
Jetson model       :
L4T version        :
CUDA version       :
llama.cpp commit   :
NGL                : 5
Context size       : 512
Maximum generation : 32 tokens
Temperature        : 0
Power mode         :
Input image size   :
```

## 24. 문제 해결

### `llama-mtmd-cli`가 생성되지 않음

CUDA 설정과 CMake 오류를 다시 확인합니다.

```bash
cd ~/vlm_lab/llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-mtmd-cli -j4
```

### `--mmproj` 또는 `--image`가 없음

멀티모달 CLI가 아닌 다른 binary를 사용했거나 소스가 맞지 않는 경우입니다.

```bash
./build/bin/llama-mtmd-cli --help | grep -E -- "--mmproj|--image"
```

소스를 갱신하면 동작과 로그 형식이 바뀔 수 있으므로 기존 실험 결과와 비교할 때 commit을 반드시 기록합니다.

### 모델 또는 projector 파일을 찾을 수 없음

```bash
find ~/vlm_lab/models -maxdepth 2 -type f -name "*.gguf" -print
```

출력된 실제 파일명과 `MODEL`, `MMPROJ` 변수가 일치하는지 확인합니다.

### `CUDA out of memory`

8GB Jetson에서는 `-ngl 99`처럼 많은 layer를 한 번에 GPU에 offload하면 CUDA memory allocation이 실패할 수 있습니다.

대표적인 오류는 다음과 같습니다.

```text
cudaMalloc failed: out of memory
failed to allocate CUDA0 buffer
unable to allocate CUDA0 buffer
```

먼저 현재 메모리 상태를 확인합니다.

```bash
free -h
tegrastats
```

필요하면 Jetson을 재부팅하여 다른 프로세스가 사용 중인 메모리를 정리합니다.

```bash
sudo reboot
```

다시 접속한 뒤 다음과 같이 보수적인 조건부터 시작합니다.

```bash
./build/bin/llama-mtmd-cli \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --image "$IMAGE" \
  -p "Describe this image in one sentence." \
  -n 32 \
  -c 512 \
  --temp 0 \
  -ngl 1 \
  --no-mmproj-offload
```

성공하면 `-ngl 1 → 5 → 10 → ...` 순서로 올립니다.

모델과 projector 조합 자체가 정상인지 확인하려면 CPU-only 실행을 사용할 수 있습니다.

```bash
./build/bin/llama-mtmd-cli \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --image "$IMAGE" \
  -p "Describe this image in one sentence." \
  -n 32 \
  -c 512 \
  --temp 0 \
  --device none \
  --no-mmproj-offload
```

CPU-only가 성공하고 CUDA 실행만 실패한다면 모델 파일 손상보다는 GPU offload 및 메모리 배치 문제를 우선 의심합니다.

### 응답 파일이 비어 있음

```bash
cat [결과_디렉터리]/runtime.log
cat [결과_디렉터리]/process_time.txt
```

모델과 mmproj 조합, 이미지 형식, 메모리 부족 및 CLI option 오류를 확인합니다. 빈 응답은 유효한 실험 결과로 사용하지 않습니다.

### `hf: command not found`

가상환경이 활성화되어 있는지 확인합니다.

```bash
source ~/vlm_lab/vlm_env/bin/activate
pip install -U huggingface_hub
hf --help
```

### 기존 결과 디렉터리가 이미 존재함

같은 `OUT_DIR`에 다시 실행하면 일부 결과가 덮어쓰일 수 있습니다. 실험마다 새로운 이름을 사용합니다.

```bash
OUT_DIR=~/vlm_lab/results/single_retry_01 bash ...
```

## 25. 카메라 기반 live VLM 확장

USB 카메라를 연결하여 실시간 영상을 입력할 수는 있지만, 현재 SmolVLM2 구성에서 모든 카메라 프레임을 VLM에 넣는 방식은 현실적이지 않습니다.

카메라가 30 FPS라면 매초 30장의 이미지가 들어오지만, VLM은 한 장의 이미지에 대해 vision encoding, prompt evaluation, text generation을 수행해야 합니다.

따라서 권장 구조는 다음과 같습니다.

```text
Camera
  ↓
가벼운 실시간 처리
YOLO / tracking / motion detection
  ↓
필요한 순간의 frame 선택
  ↓
SmolVLM2
  ↓
장면 설명 또는 질의응답
```

또는 간단한 실습에서는 일정 간격으로 한 장만 선택합니다.

```text
Camera 30 FPS
  ↓
예: 5초마다 frame 1장 선택
  ↓
VLM 분석
```

실제 live application에서는 매 frame마다 `llama-mtmd-cli` process를 새로 실행하는 방식보다 모델을 메모리에 유지하는 persistent process 또는 server 구조가 적합합니다.

본 실습에서 사용하는 `run_vlm.sh`는 의도적으로 매 질문마다 새 process를 시작하므로 다음을 측정합니다.

```text
cold-process cost
= process startup
+ model load
+ mmproj load
+ vision encoding
+ prompt evaluation
+ decoding
```

따라서 본 실습 결과를 그대로 실시간 camera FPS로 해석하지 않습니다.

## 참고 자료

- [llama.cpp multimodal/mtmd 공식 문서](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md)
- [llama-mtmd-cli 공식 소스 및 사용법](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/mtmd-cli.cpp)
- [SmolVLM2-2.2B-Instruct Q4_K_M GGUF](https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/blob/main/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf)
- [SmolVLM2-2.2B-Instruct Q8_0 mmproj](https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/blob/main/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf)
