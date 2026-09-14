# Jetson 접속 및 종료 방법

Jetson 실습을 시작할 때는 **PowerShell 또는 VS Code Remote SSH**로 접속하고, 실습 종료 시에는 반드시 Jetson을 정상 종료한다.

---

## 1. PowerShell로 Jetson 접속

Windows PowerShell을 실행한 뒤 Jetson에 SSH로 접속한다.

```powershell
ssh hansungai@<JETSON_IP>
```

예:

```powershell
ssh hansungai@192.168.0.100
```

처음 접속하는 경우 다음과 같은 메시지가 나타날 수 있다.

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

이 경우:

```text
yes
```

를 입력한다.

이후 Jetson 계정 비밀번호를 입력한다.

정상적으로 접속되면 다음과 비슷한 프롬프트가 나타난다.

```text
hansungai@ubuntu:~$
```

---

## 2. 실습 폴더로 이동

예를 들어 Quantization 실습의 경우:

```bash
cd ~/quantization_lab
```

가상환경 활성화:

```bash
source quant_env/bin/activate
```

정상적으로 활성화되면:

```text
(quant_env) hansungai@ubuntu:~/quantization_lab$
```

와 같이 표시된다.

---

## 3. PowerShell SSH 접속 종료

Jetson은 계속 켜두고 SSH 연결만 종료하려면:

```bash
exit
```

또는

```text
Ctrl + D
```

를 사용한다.

---

## 4. Jetson 전원 종료

실습이 끝난 뒤 Jetson의 전원을 끌 때는 전원 케이블을 바로 뽑지 말고 반드시 정상 종료한다.

```bash
sudo shutdown -h now
```

또는

```bash
sudo poweroff
```

비밀번호 입력 후 잠시 기다리면 Jetson이 종료된다.

> **주의**
>
> `exit`는 SSH 세션만 종료한다.
>
> ```bash
> exit
> ```
>
> `sudo shutdown -h now`는 Jetson 자체를 종료한다.
>
> ```bash
> sudo shutdown -h now
> ```

---

## 5. VS Code Remote SSH로 접속

### Step 1. VS Code 실행

Windows에서 VS Code를 실행한다.

Remote SSH Extension이 설치되어 있어야 한다.

Extension 이름:

```text
Remote - SSH
```

### Step 2. Remote SSH 실행

VS Code 왼쪽 아래의 Remote 버튼을 클릭하거나:

```text
Ctrl + Shift + P
```

를 눌러 Command Palette를 연다.

다음을 검색한다.

```text
Remote-SSH: Connect to Host...
```

### Step 3. Jetson 선택

등록된 Jetson SSH 주소를 선택한다.

예:

```text
hansungai@192.168.0.100
```

등록되어 있지 않은 경우:

```text
Remote-SSH: Add New SSH Host...
```

를 선택한 뒤 다음을 입력한다.

```bash
ssh hansungai@<JETSON_IP>
```

### Step 4. Jetson 폴더 열기

Remote SSH 연결 후:

```text
File
→ Open Folder
```

예를 들어:

```text
/home/hansungai/quantization_lab
```

을 선택한다.

VS Code Terminal을 열면 Jetson의 terminal이 실행된다.

```text
Terminal
→ New Terminal
```

### Step 5. 가상환경 활성화

VS Code Terminal에서:

```bash
source quant_env/bin/activate
```

예:

```text
(quant_env) hansungai@ubuntu:~/quantization_lab$
```

---

## 6. VS Code에서 실습 종료

실습 종료 시 VS Code Terminal에서 필요한 프로그램을 먼저 종료한다.

실행 중인 Python 프로그램 등이 있다면:

```text
Ctrl + C
```

가상환경 종료는 선택 사항이다.

```bash
deactivate
```

---

## 7. VS Code에서 Jetson 종료

VS Code Terminal에서 다음 명령을 실행한다.

```bash
sudo shutdown -h now
```

또는

```bash
sudo poweroff
```

Jetson이 종료되면 Remote SSH 연결도 자동으로 끊긴다.

---

## 8. 매 실습 시 기본 순서

### PowerShell 사용

```text
PowerShell 실행
    ↓
ssh hansungai@<JETSON_IP>
    ↓
cd 실습폴더
    ↓
source 가상환경/bin/activate
    ↓
실습
    ↓
sudo shutdown -h now
```

예:

```powershell
ssh hansungai@192.168.0.100
```

```bash
cd ~/quantization_lab
source quant_env/bin/activate

# 실습 진행

sudo shutdown -h now
```

### VS Code 사용

```text
VS Code 실행
    ↓
Remote-SSH: Connect to Host
    ↓
Jetson 선택
    ↓
Open Folder
    ↓
Terminal 실행
    ↓
source 가상환경/bin/activate
    ↓
실습
    ↓
sudo shutdown -h now
```

---

## 9. 자주 사용하는 명령어

| 목적 | 명령어 |
|---|---|
| SSH 접속 | `ssh hansungai@<JETSON_IP>` |
| 현재 위치 확인 | `pwd` |
| 파일 목록 확인 | `ls` |
| 실습 폴더 이동 | `cd ~/quantization_lab` |
| 가상환경 활성화 | `source quant_env/bin/activate` |
| 가상환경 종료 | `deactivate` |
| SSH 연결만 종료 | `exit` |
| Jetson 정상 종료 | `sudo shutdown -h now` |
| Jetson 즉시 종료 | `sudo poweroff` |
| Jetson 재부팅 | `sudo reboot` |

---

## 10. 주의사항

- Jetson 사용 중 전원 케이블을 바로 분리하지 않는다.
- 반드시 `sudo shutdown -h now` 또는 `sudo poweroff`로 종료한다.
- `exit`는 Jetson 종료가 아니라 SSH 연결 종료이다.
- VS Code Remote SSH를 닫는 것만으로 Jetson 전원이 꺼지지는 않는다.
- 여러 학생이 Jetson을 사용할 경우 실습 종료 후 정상 종료 여부를 반드시 확인한다.
