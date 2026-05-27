# STT (WhisperX) - 개발 표준

문서 버전: 1.0
대상 독자: 백엔드/AI 개발자, DevOps
관련 문서: [`./development-environment.md`](./development-environment.md)

---

## 목차

1. 개요
2. 개발환경
   2.1 개발환경 구성도
   2.2 개발절차
   2.3 개발자 PC 구성 내역
   2.4 IDE (Cursor / VSCode / PyCharm)
   2.5 소스 관리 (사내 Git + GitHub 미러)
   2.6 모델 / 패키지 / 이미지 저장소
   2.7 IDE 설정 및 런타임 설치
       2.7.1 IDE 설정 (Cursor / VSCode)
       2.7.2 Python / uv 설치
       2.7.3 CUDA 12.x / NVIDIA Container Toolkit
       2.7.4 시스템 의존성 (ffmpeg / git)
       2.7.5 Docker / Compose
       2.7.6 Whisper / Alignment / Pyannote 모델 배치
       2.7.7 Hugging Face 인증
3. 디렉토리 & 모듈 표준
4. 의존성 / 패키지 관리 표준
5. 코드 스타일 표준
6. API 표준
7. 모델 운영 규칙
8. Job 관리
9. Docker / 배포 표준
10. 로깅 / 관측
11. 보안 / 데이터
12. Git / 브랜치 / PR
13. 백엔드 연동 시 주의

---

## 1. 개요

본 문서는 STT 서비스(`/home/pps-nipa/jenkins/dev/stt`)의 **개발 환경 / 모델 / 코드 / 배포** 표준을 정의합니다.
서비스는 **WhisperX** 기반으로 음성 → 텍스트(ASR) + **강제 정렬**(forced alignment) + **화자 분리**(diarization) 를 통합 제공하는 **비동기 Job 기반 API** 입니다.

| 구분 | 기술 |
|------|------|
| 언어 | Python ≥ 3.10 |
| API | FastAPI + Uvicorn + BackgroundTasks |
| ASR | faster-whisper (CTranslate2) — Whisper `large-v3` 기본 |
| Diarization | pyannote-audio 3.x |
| Alignment | Wav2Vec2 (Hugging Face) |
| 추론 백엔드 | PyTorch 2.5.1 + CUDA 12.1 |
| 패키지 매니저 | uv |
| 컨테이너 | Docker / docker compose |
| CI | Jenkins (`/home/pps-nipa/jenkins/`) |

---

## 2. 개발환경

### 2.1 개발환경 구성도

```
┌──────────────────────────────────────────────────────────────────┐
│                          개발자 PC                                │
│   Cursor IDE  ──────────  Python 3.10 + uv (.venv)                │
│        │                        │                                 │
│        │ SSH/HTTPS              │ docker (GPU)                    │
└────────┼────────────────────────┼─────────────────────────────────┘
         │                        │
         ▼                        ▼
┌────────────────────┐    ┌───────────────────────────────────────┐
│  사내 Git (Gitea)   │    │  Hugging Face Hub (pyannote, w2v2)    │
│  narea/stt.git     │    │  /  사내 NAS 미러                       │
└────────┬───────────┘    └───────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌────────────────────┐         ┌───────────────────────────────────┐
│ GitHub 미러         │         │ src/resources/models/             │
│ GiJeongCho/whisperX │         │  ├── whisper/                     │
└────────────────────┘         │  ├── alignment/<lang>/             │
         │                      │  ├── diarization/                  │
         │                      │  ├── vad/                          │
         │                      │  └── embedding/                    │
         │                      └───────────────────────────────────┘
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                Jenkins 서버 (Build / Deploy)                      │
│  dev/docker.sh dev up stt_api                                     │
│        │                                                          │
│        ▼                                                          │
│   ┌────────────────────────────────────────┐                      │
│   │ stt_api (FastAPI + BG Tasks)            │ ◄── /transcribe     │
│   │ Whisper + Align + Diarize on GPU        │     /jobs/{id}      │
│   └────────────────────────────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 개발절차

1. 개발자 PC에 IDE, Python 3.10, uv, Docker, NVIDIA 드라이버 설치.
2. SSH 키 등록(사내 Git, GitHub), Hugging Face 토큰 발급(pyannote 등 gated 모델).
3. `git clone ssh://git@git.biz.ppsystem.co.kr:10022/narea/stt.git`.
4. `uv sync`.
5. `ffmpeg` 설치 (`apt-get install ffmpeg`).
6. 모델 다운로드:
   ```bash
   python src/resources/download_models.py --output ./src/resources/models
   ```
7. `uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload` → `/health` 확인.
8. PR → 사내 Git push → GitHub 미러(`GiJeongCho/whisperX`) 동시 반영.
9. Jenkins Job 트리거 → dev/stg/prd 배포 → 스모크 테스트.
10. SLA 측정(짧은 / 긴 음성 각각).

### 2.3 개발자 PC 구성 내역

| 항목 | 최소 | 권장 | 비고 |
|------|------|------|------|
| OS | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS | |
| CPU | 8 core | 16 core+ | VAD/디아리제이션 전처리에 CPU 부담 |
| RAM | 32 GB | 64 GB | 긴 음성 처리 시 메모리 |
| Disk | 100 GB | 500 GB SSD | 모델 합계 ≈ 5~8 GB |
| GPU | RTX 3060 12GB+ | RTX 4090 / A100 | CUDA 12.x |
| Python | 3.10 / 3.11 | 3.10.x | |
| Docker | 24.x | 26.x | `--gpus all` |
| ffmpeg | 4.x | 6.x | |

### 2.4 IDE (Cursor / VSCode / PyCharm)

- 권장: Cursor 또는 VSCode.
- 필수 확장:
  - **Python**, **Pylance**, **Ruff**
  - **Docker**
  - **REST Client** 또는 **Thunder Client**
  - **Audio Preview** (`audio-preview` 류) — 샘플 wav 확인용
  - **Even Better TOML**

### 2.5 소스 관리 (사내 Git + GitHub 미러)

- 사내 Git: `ssh://git@git.biz.ppsystem.co.kr:10022/narea/stt.git`
- GitHub 미러: `https://github.com/GiJeongCho/whisperX.git`
- `origin` 에 fetch 1 + push 2. `git push origin <branch>` 한 번으로 동시 반영.

### 2.6 모델 / 패키지 / 이미지 저장소

| 자원 | 저장소 | 비고 |
|------|--------|------|
| Whisper (faster-whisper) | Hugging Face Hub `Systran/faster-whisper-large-v3` 등 / 사내 NAS 미러 | offline 운영 시 NAS 미러 |
| pyannote 화자 분리 | HF Hub (gated) | `HF_TOKEN` 필요 |
| Wav2Vec2 정렬 모델 | HF Hub | 언어별로 별도 |
| Python 패키지 | PyPI / 사내 Nexus | `pytorch` cu128 인덱스 사용 |
| Docker 이미지 | 사내 Registry (`IMG_STT_API`) | dev/stg/prd 태그 분리 |

### 2.7 IDE 설정 및 런타임 설치

#### 2.7.1 IDE 설정 (Cursor / VSCode)

`.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.tabSize": 4
  },
  "files.watcherExclude": {
    "**/src/resources/models/**": true,
    "**/.venv/**": true
  }
}
```

`.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "STT API (uvicorn)",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": ["src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
      "env": {
        "WHISPER_MODEL_DIR": "${workspaceFolder}/src/resources/models",
        "WHISPER_ARCH": "large-v3",
        "PYTHONPATH": "${workspaceFolder}"
      },
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

#### 2.7.2 Python / uv 설치

```bash
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /home/pps-nipa/jenkins/dev/stt
uv sync
```

#### 2.7.3 CUDA 12.x / NVIDIA Container Toolkit

```bash
nvidia-smi   # CUDA 12.x 호환
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 2.7.4 시스템 의존성 (ffmpeg / git)

```bash
sudo apt-get install -y ffmpeg git curl
```

#### 2.7.5 Docker / Compose

```bash
curl -fsSL https://get.docker.com | sh
docker compose version
```

#### 2.7.6 Whisper / Alignment / Pyannote 모델 배치

표준 구조:
```
src/resources/models/
├── whisper/                       # faster-whisper (CTranslate2)
│   ├── model.bin
│   ├── config.json
│   └── tokenizer.json
├── alignment/
│   ├── en/                        # wav2vec2 영어 정렬
│   └── ko/                        # wav2vec2 한국어 정렬
├── diarization/                   # pyannote/speaker-diarization-3.x
├── vad/                           # pyannote VAD
└── embedding/                     # pyannote embedding
```

다운로드 헬퍼:
```bash
# 기본 large-v3 + ko/en alignment + pyannote-3.x
HF_TOKEN=hf_xxxx python src/resources/download_models.py \
  --output ./src/resources/models
```

#### 2.7.7 Hugging Face 인증

```bash
huggingface-cli login
# Jenkins 측에서는 Credentials 'HF_TOKEN' 으로 등록 → model_download.sh 가 사용
```

---

## 3. 디렉토리 & 모듈 표준

| 레이어 | 위치 | 책임 |
|--------|------|------|
| API | `src/api.py` | FastAPI 라우팅, 업로드, BackgroundTasks, Job 상태 |
| 서비스 | `src/service.py` | `WhisperXService` (모델 로드, transcribe 파이프라인) |
| 유틸 | `src/v1/utils/` | KST 시각, torch.load 패치 |
| 자원 | `src/resources/` | 모델 가중치, 다운로드 스크립트 |
| 테스트 | `src/test/` | 음성 샘플, 응답 스냅샷, `test_whisperx.py` |

> **금지**: API 핸들러에서 `whisperx` 를 직접 호출하지 않는다. 반드시 `WhisperXService` 를 통한다.

### 3.1 API 버저닝
- 현재 단일 라우트(`/transcribe`, `/jobs/{id}`). 입출력 스키마 변경 시 `/v2/transcribe` 신설 + Pydantic 모델 분리.
- 유틸은 `src/v1/utils/` 에 모음. 신규 버전은 `src/v2/utils/`.

### 3.2 네이밍
- 모듈: `snake_case`
- 클래스: `PascalCase` (`WhisperXService`)
- 상수/환경변수: `UPPER_SNAKE`

---

## 4. 의존성 / 패키지 관리 표준

- 로컬: `uv` 우선. `uv sync` 로 재현.
- Docker: `requirements.txt` 의 **git 핀 + 버전 핀** 유지.
- 추가 시:
  1. `pyproject.toml` 에 추가 + 인덱스 명시.
  2. `requirements.txt` 동기 갱신.
  3. PyTorch/CUDA 변경은 base image 변경과 한 PR로.

핵심 핀:
```
whisperx @ git+https://github.com/m-bain/whisperX.git@<commit>
torch (cu128 index)
pyannote-audio>=3.3.2,<4.0.0
ctranslate2>=4.5.0
faster-whisper>=1.1.1
```

---

## 5. 코드 스타일 표준

- PEP8, 4-space, 100자.
- docstring 필수, 특히 `transcribe_audio` 같은 핵심 메서드.
- 타입 힌트 의무.
- `print` 금지 → `logger`.

### 5.1 로깅
- 모듈별 `logger = logging.getLogger(__name__)`.
- **Job ID 포함**: `logger.info(f"[Job {job_id}] ...")` 패턴 유지.
- 모델 디렉토리 점검 로그(`_log_directory_contents`)는 lifespan 단계에서 유지 — 운영 디버깅 핵심.

### 5.2 예외 처리
- 모델 로드 실패는 lifespan try/except 에서 잡고 `/health` 로 노출. 컨테이너는 죽이지 않음.
- 추론 실패 → `JobInfo.status="failed"`, `progress.step="failed_at_<step>"`, `progress.percent=-1`.
- 임시파일은 `finally` 에서 삭제.

---

## 6. API 표준

### 6.1 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 모델/디바이스 상태 |
| POST | `/transcribe` (202) | Job 생성. 즉시 `{job_id, message}` 반환 |
| GET | `/jobs/{job_id}` | Job 상태/진행률/결과 |

### 6.2 응답 표준
- Job 결과 `result` 는:
  - `segments`: `[{start, end, text, words?, speaker?}]`
  - `language`: 추론된 언어 코드
  - `processing_time`: 초
  - `meta`: 모델/파라미터 메타
- 시간 필드는 **KST ISO 8601** (`get_kst_now_iso()`).

### 6.3 진행률
- `progress = {"step": <name>, "percent": <int>}`.
- 표준 step: `starting`, `asr`, `align`, `diarize`, `merge`, `done`.
- 실패 시 `failed_at_<step>` + `-1`.

### 6.4 파라미터 검증
- 화자 수 0 입력은 `None` 으로 정규화.
- `suppress_tokens` 는 콤마 구분 문자열 → `list[int]`.
- 새 파라미터 추가 시 `Form(..., description=...)` 로 설명 명시.

---

## 7. 모델 운영 규칙

### 7.1 모델 로딩 (lifespan)
- Whisper, Alignment(ko/en pre-load), Diarization, VAD, Embedding 모두 로컬 경로에서 로드 → **오프라인 보장**.
- 디바이스: `cuda` 가능 시 `float16`, 아니면 `int8`.
- `WHISPER_MODEL_DIR` 하위 표준 구조 강제(2.7.6 참조).

### 7.2 Diarization
- `pyannote-audio` 3.x 만 지원 (`pyproject` 의 `<4.0` 제약).
- 오프라인 운영 시: 가중치를 `models/diarization`에 사전 배치하고 코드의 `local` 로드 경로 사용.
- SPEAKER 라벨(`SPEAKER_00`, ...) 은 본 서비스가 부여. 사람 이름 매핑은 **speech_recognize 서비스** 책임.

### 7.3 Alignment
- 사전 로드 언어: `["en", "ko"]`. 필요 언어는 lifespan 코드에 추가.
- 가중치 누락 시 warning 후 계속 진행(해당 언어만 align 비활성).

---

## 8. Job 관리

- 현재는 **메모리 dict (`jobs`)** 기반. 단일 프로세스/단일 워커 가정.
- 다중 워커(`uvicorn --workers > 1`) 사용 금지. 필요 시 Redis 기반 잡 저장소로 이행 PR.
- Job 결과는 영구 저장하지 않음. 백엔드는 **완료 즉시 결과를 pull 하여 자체 DB에 저장** 해야 한다.

---

## 9. Docker / 배포 표준

- Base image: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`.
- `ENV WHISPER_MODEL_DIR=/app/src/resources/models`, `APP_PORT=6002`.
- 모델은 이미지 미포함 + 볼륨 마운트.
- `.dockerignore`: `src/resources/models/`, `src/test/*.wav`, `.venv/`, `nohup.out`, `server.log`, `__pycache__/`.

### 9.1 Jenkins 연동
- `IMG_STT_API=<registry>/stt-api:<env>`.
- compose 서비스명: `stt_api`. 내부 URL: `STT_API_BASE=http://stt_api:8000`.
- 포트: dev 6002 / stg 9002 / prd 8002.

### 9.2 헬스체크
```yaml
healthcheck:
  test: ["CMD", "curl", "-fsS", "http://localhost:${APP_PORT}/health"]
  interval: 30s
  timeout: 5s
  retries: 5
```

---

## 10. 로깅 / 관측

| 항목 | 표준 |
|------|------|
| 요청 로그 | `[Job {id}] Processing {filename}` |
| 진행 로그 | `step` / `percent` |
| 완료 로그 | `Completed in Ns` |
| 실패 로그 | `logger.error(..., exc_info=True)` + `progress=failed_at_*` |
| `/health` | 항상 200. payload 로 모델 상태 |

---

## 11. 보안 / 데이터

- 업로드 음성 파일은 임시 디렉토리에 저장 후, 백그라운드 작업 종료 시점에 **반드시 삭제** (`background_process` 의 finally 절 유지).
- 응답에 포함되는 텍스트는 민감정보일 수 있음 → 백엔드는 로그/저장 정책에 맞게 처리.
- `HF_TOKEN` 등 시크릿은 코드/이미지에 포함하지 않고 환경변수로만 전달.
- CORS 와일드카드는 dev/stg 한정. prd 에서는 ingress(nginx) 단에서 도메인 화이트리스트.

---

## 12. Git / 브랜치 / PR

- 브랜치: `feat/stt-<topic>`, `fix/stt-<topic>`.
- 커밋: `[stt] <동사> <내용>`.
- 모델/오디오 파일 커밋 금지(`*.bin`, `*.wav`, `*.pt`, `*.onnx`).
- WhisperX git pin 변경 시 PR 본문에 **벤치(언어별 WER, 처리 속도)** 첨부 권장.
- 사내 Git + GitHub 미러(`GiJeongCho/whisperX`) 동시 push.

---

## 13. 백엔드 연동 시 주의

- **항상 비동기 Job 패턴**. 동기 응답을 기대하지 말 것.
- 큰 파일(>30분)은 GPU 1장당 동시 1~2건만 권장. 백엔드 큐로 직렬화.
- 결과는 단명(in-memory). **백엔드가 폴링 완료 즉시 가져가서 자체 저장.**
- 화자 라벨은 익명(`SPEAKER_00`). 사람 매핑은 **speech_recognize** 호출 단계에서 수행.
- 호출 순서 표준: `stt → speech_recognize → llm (요약 등)`.
