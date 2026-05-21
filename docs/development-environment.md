# STT (WhisperX) - 개발 환경 가이드

> 백엔드 개발자 대상 문서.
> 본 서비스는 **WhisperX** 를 기반으로 한 **온프레미스 STT(Speech-to-Text) + 강제 정렬(Alignment) + 화자 분리(Diarization)** 통합 추론 API입니다.
> 화자 식별(직원 매칭)은 별도 `speech_recognize` 서비스가 담당합니다.

---

## 1. 서비스 개요

| 항목 | 내용 |
|------|------|
| 서비스 이름 | WhisperX API |
| 코드상 클래스 | `WhisperXService` (`src/service.py`) |
| 모델 | OpenAI Whisper(`large-v3` 기본) + Pyannote Diarization + Wav2Vec2 Alignment |
| 처리 단계 | 1) ASR(Whisper) 2) Forced Alignment 3) Speaker Diarization(`pyannote-audio`) |
| 처리 방식 | **비동기 Job 패턴** (`POST /transcribe` → `job_id` → `GET /jobs/{job_id}`) |
| 디바이스 | GPU (CUDA) 권장. compute_type = `float16` (GPU) / `int8` (CPU) |

---

## 2. 기술 스택 (AI / Framework)

### 2.1 런타임
- **Python**: `>=3.10` (`pyproject.toml`)
- **Docker base**: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
- **CUDA**: 12.1, cuDNN 9 (런타임). PyTorch wheel은 cu128 인덱스 사용
- **OS 의존성**: `ffmpeg`, `git`, `curl`

### 2.2 핵심 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| `whisperx` | git tag `646f511...` | ASR + alignment + diarization 통합 파이프라인 |
| `faster-whisper` | ≥ 1.1.1 | Whisper 추론(CTranslate2 백엔드) |
| `ctranslate2` | ≥ 4.5 | 최적화된 추론 엔진 |
| `pyannote-audio` | ≥ 3.3.2, < 4.0 | 화자 분리 |
| `huggingface-hub` | < 1.0 | 모델 다운로드/캐시 |
| `transformers` | ≥ 4.48 | Wav2Vec2 정렬 모델 |
| `torch`, `torchaudio` | cu128 인덱스 | 추론 백엔드 |
| `nltk` | ≥ 3.9 | 텍스트 후처리 |
| `triton` | ≥ 3.3 (Linux x86_64) | 일부 커널 가속 |
| `pandas`, `numpy` | - | 후처리 |
| `fastapi`, `uvicorn`, `python-multipart` | pin in `requirements.txt` | API 서버 |

### 2.3 패키지 매니저
- 로컬: **uv** (`uv.lock`, `pyproject.toml`)
- Docker: `requirements.txt` 의 git 의존성 + 핀 버전
  ```
  whisperx @ git+https://github.com/m-bain/whisperX.git@646f511...
  torchvision==0.23.0
  fastapi==0.135.1
  uvicorn==0.42.0
  python-multipart==0.0.22
  ```

---

## 3. 디렉토리 구조

```
stt/
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── README.md
├── src/
│   ├── api.py                      # FastAPI 엔트리포인트 (/transcribe, /jobs/{id}, /health)
│   ├── service.py                  # WhisperXService (모델 로드/추론)
│   ├── test_whisperx.py            # 로컬 추론 스모크 테스트
│   ├── v1/utils/
│   │   ├── time_utils.py           # KST 시간 헬퍼
│   │   └── torch_utils.py          # torch.load 패치
│   ├── resources/
│   │   ├── download_models.py      # 모델 사전 다운로드 스크립트
│   │   └── models/
│   │       ├── whisper/            # Whisper (faster-whisper) 가중치 (model.bin 등)
│   │       ├── alignment/          # 언어별 정렬 모델
│   │       ├── diarization/        # pyannote 화자 분리
│   │       ├── vad/                # VAD 모델
│   │       └── embedding/          # 화자 임베딩
│   └── test/                       # 샘플 오디오 + 응답 JSON
├── whisperX/                       # (git submodule placeholder; 현재 비어있음)
└── docs/                           # 본 문서 위치
```

---

## 4. 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `WHISPER_MODEL_DIR` | `src/resources/models` (Docker: `/app/src/resources/models`) | 모델 루트 디렉토리 |
| `WHISPER_ARCH` | `large-v3` | Whisper 아키텍처(미리 다운로드된 가중치 없으면 이 이름으로 폴백) |
| `APP_PORT` | 6002 (Docker default) / 8000 (README) | 서버 포트 |
| `HF_TOKEN` | (선택) | Pyannote 모델 최초 다운로드용 (오프라인 운영 시 불필요) |

> **포트 규칙**: dev = 6002 / stg = 9002 / prd = 8002. compose 서비스명 `stt_api`, 내부 URL = `STT_API_BASE=http://stt_api:8000`.

---

## 5. 로컬 개발 환경 구축

### 5.1 사전 요구사항
- NVIDIA GPU + 드라이버
- CUDA 12.x 호환
- `ffmpeg`
- `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get install -y ffmpeg
```

### 5.2 설치

```bash
cd /home/pps-nipa/jenkins/dev/stt

# 의존성 설치
uv sync

# (인터넷 가능 환경에서 1회) 모델 다운로드
python src/resources/download_models.py --output ./src/resources/models
# 또는 HuggingFace Cache 를 사용한 후, 위 디렉토리 구조로 복사

# 서버 실행 (개발)
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### 5.3 헬스체크
```bash
curl http://localhost:8000/health
```
응답:
```json
{
  "status": "ok",
  "models": {
    "whisper_loaded": true,
    "alignment_loaded": ["ko", "en"],
    "diarization_loaded": true,
    "device": "cuda",
    "compute_type": "float16"
  }
}
```

---

## 6. Docker 실행

### 6.1 단독 실행
```bash
cd /home/pps-nipa/jenkins/dev/stt
docker build -t pps/stt-api:dev .

docker run --rm \
  --gpus all \
  -e APP_PORT=6002 \
  -p 6002:6002 \
  -v $(pwd)/src/resources/models:/app/src/resources/models:ro \
  pps/stt-api:dev
```

### 6.2 Jenkins 통합 배포
- 이미지 태그: `IMG_STT_API=<registry>/stt-api:<env>`
- compose 서비스명: `stt_api`
- 모델 경로: 호스트 → 컨테이너 read-only 마운트

```bash
sudo /home/pps-nipa/jenkins/dev/docker.sh dev up stt_api
```

---

## 7. API 사용법

### 7.1 비동기 전사 요청

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@meeting.wav" \
  -F "language=ko" \
  -F "align=true" \
  -F "diarize=true" \
  -F "min_speakers=2" \
  -F "max_speakers=5"
# -> {"job_id": "<uuid>", "message": "Job submitted"}
```

주요 파라미터:

| 필드 | 기본 | 설명 |
|------|------|------|
| `language` | `ko` | 언어 코드 (`ko`, `en`, ...) |
| `batch_size` | 16 | 디코딩 배치 |
| `beam_size` | 5 | 빔 탐색 |
| `temperature` | 0.0 | 샘플링 온도(0=greedy) |
| `compression_ratio_threshold` | 2.4 | 잡음 컷오프 |
| `log_prob_threshold` | -1.0 | 평균 로그확률 컷오프 |
| `no_speech_threshold` | 0.6 | 묵음 감지 |
| `condition_on_previous_text` | false | 이전 텍스트 문맥 사용 |
| `initial_prompt` | null | 초기 프롬프트 |
| `suppress_tokens` | `-1` | 억제 토큰 ID (콤마 구분) |
| `align` | true | 단어 단위 강제 정렬 |
| `diarize` | true | 화자 분리 |
| `min_speakers` / `max_speakers` | null | 화자 수 힌트 |
| `vad_onset` / `vad_offset` | 0.5 / 0.363 | VAD 임계값 |

### 7.2 Job 상태 조회

```bash
curl "http://localhost:8000/jobs/<uuid>"
```

응답:
```json
{
  "status": "completed",
  "created_at": "2026-05-21T15:30:00+09:00",
  "started_at": "...",
  "completed_at": "...",
  "progress": {"step": "done", "percent": 100},
  "filename": "meeting.wav",
  "result": {
    "segments": [
      {"start": 0.0, "end": 4.2, "text": "...", "speaker": "SPEAKER_00", "words": [...]}
    ],
    "language": "ko",
    "processing_time": 12.3,
    "meta": {...}
  }
}
```

### 7.3 Swagger
- `http://localhost:8000/docs`

---

## 8. 모델 디렉토리 규약

```
src/resources/models/
├── whisper/
│   ├── model.bin
│   ├── config.json
│   └── tokenizer.json
├── alignment/
│   └── <hf_cache 형태로 언어별 wav2vec2>
├── diarization/
│   └── <pyannote/speaker-diarization-3.x>
├── vad/
│   └── <pyannote VAD>
└── embedding/
    └── <pyannote embedding>
```

- 모델 파일이 누락되면 lifespan 시점에 에러 로그가 남고 해당 단계는 비활성화됩니다(`/health` 에서 확인 가능).

---

## 9. 트러블슈팅

| 증상 | 원인 | 조치 |
|------|------|------|
| `503 Models not loaded` | `WhisperXService.load_models()` 실패 | 컨테이너 로그 확인 → 모델 디렉토리 누락 점검 |
| Diarization 실패 | pyannote 가중치 미설치, HF 토큰 미설정 | 오프라인은 가중치 마운트 / 온라인은 `HF_TOKEN` 설정 |
| GPU 미사용 | CUDA 미설치 또는 `--gpus all` 누락 | `nvidia-smi`, `torch.cuda.is_available()` 검증 |
| 정렬(alignment) 누락 언어 | wav2vec2 가중치 부재 | `alignment/<lang>/` 추가 |
| `triton` 설치 실패 (macOS/ARM) | 플랫폼 미지원 | `pyproject.toml` 의 marker 조건이 자동 스킵 — 무시 가능 |

---

## 10. 관련 문서
- 개발 표준 → [`./development-standards.md`](./development-standards.md)
- 화자 식별 → [`/home/pps-nipa/jenkins/dev/speech_recognize/docs/development-environment.md`](../../speech_recognize/docs/development-environment.md)
- Jenkins 배포 → [`/home/pps-nipa/jenkins/docs/development-environment.md`](../../../docs/development-environment.md)
