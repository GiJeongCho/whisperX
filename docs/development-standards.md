# STT (WhisperX) - 개발 표준

> WhisperX 기반 STT API의 코드/구조/운영 표준.
> 모델 버전 변경, 비동기 처리 변경, 진행률/Job 인터페이스 변경 시 본 문서를 우선 참조합니다.

---

## 1. 디렉토리 & 모듈 규칙

| 레이어 | 위치 | 책임 |
|--------|------|------|
| API | `src/api.py` | FastAPI 라우팅, 업로드, BackgroundTasks, Job 상태 |
| 서비스 | `src/service.py` | `WhisperXService` (모델 로드, transcribe 파이프라인) |
| 유틸 | `src/v1/utils/` | KST 시각, torch.load 패치 |
| 자원 | `src/resources/` | 모델 가중치, 다운로드 스크립트 |
| 테스트 | `src/test/` | 음성 샘플, 응답 스냅샷, `test_whisperx.py` |

> **금지**: API 핸들러 안에서 `whisperx` 를 직접 호출하지 않는다. 반드시 `WhisperXService` 를 통한다.

### 1.1 API 버저닝
- 현재 단일 라우트(`/transcribe`, `/jobs/{id}`). 입출력 스키마 변경 시 `/v2/transcribe` 신설 + Pydantic 모델 분리.
- 유틸은 `src/v1/utils/` 에 모음. 신규 버전은 `src/v2/utils/` 로.

### 1.2 네이밍
- 모듈: `snake_case`
- 클래스: `PascalCase` (`WhisperXService`)
- 상수/환경변수: `UPPER_SNAKE`

---

## 2. 의존성 / 패키지 관리

- 로컬: **`uv`** 우선. `uv sync` 로 재현.
- Docker: `requirements.txt` 의 **git 핀 + 버전 핀** 유지.
- 추가 시:
  1. `pyproject.toml` 의존성 추가 (uv 인덱스 명시).
  2. `requirements.txt` 도 동기 갱신.
  3. PyTorch/CUDA 버전 변경은 base image 변경과 한 PR로.

---

## 3. 코드 스타일

- PEP8, 4-space, 100자.
- 함수/클래스 docstring 필수. 특히 `transcribe_audio` 같은 핵심 메서드.
- 타입 힌트 의무.
- `print` 금지 → `logger`.

### 3.1 로깅
- 모듈별 `logger = logging.getLogger(__name__)`.
- **Job ID 포함**: `logger.info(f"[Job {job_id}] ...")` 패턴 유지.
- 모델 디렉토리 점검 로그(`_log_directory_contents`)는 lifespan 단계에서 유지 — 운영 디버깅 핵심.

### 3.2 예외 처리
- 모델 로드 실패는 lifespan 의 try/except 에서 잡고 `/health` 로 노출. 컨테이너는 죽이지 않음.
- 추론 실패 → `JobInfo.status="failed"`, `progress.step="failed_at_<step>"`, `progress.percent=-1`.
- 임시파일은 `finally` 에서 삭제.

---

## 4. API 표준

### 4.1 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 모델/디바이스 상태 |
| POST | `/transcribe` (202) | Job 생성. 즉시 `{job_id, message}` 반환 |
| GET | `/jobs/{job_id}` | Job 상태/진행률/결과 |

### 4.2 응답 표준
- Job 결과 `result` 는 다음을 포함:
  - `segments`: `[{start, end, text, words?, speaker?}]`
  - `language`: 추론된 언어 코드
  - `processing_time`: 초 단위
  - `meta`: 모델/파라미터 메타
- 시간 필드는 모두 **KST ISO 8601** (`get_kst_now_iso()` 사용).

### 4.3 진행률
- `progress = {"step": <name>, "percent": <int>}`.
- 표준 step 명칭: `starting`, `asr`, `align`, `diarize`, `merge`, `done`.
- 실패 시 `failed_at_<step>` + `-1`.

### 4.4 파라미터 검증
- 화자 수 0 입력은 `None` 으로 정규화 (현재 패턴 유지).
- `suppress_tokens` 는 콤마 구분 문자열 → `list[int]` 변환.
- 새 파라미터 추가 시 **`Form(..., description=...)`** 로 설명 명시.

---

## 5. 모델 운영 규칙

### 5.1 모델 로딩 (lifespan)
- Whisper, Alignment(ko/en pre-load), Diarization, VAD, Embedding 모두 로컬 경로에서 로드 → **오프라인 보장**.
- 디바이스: `cuda` 가능 시 `float16`, 아니면 `int8`.
- `WHISPER_MODEL_DIR` 하위 표준 구조(`whisper/`, `alignment/`, `diarization/`, `vad/`, `embedding/`) 강제.

### 5.2 Diarization
- `pyannote-audio` 3.x 만 지원 (`pyproject` 의 `<4.0` 제약).
- 오프라인 운영 시: 가중치를 `models/diarization`에 사전 배치하고 코드의 `local` 로드 경로 사용.
- 다이얼라이저션 결과의 SPEAKER 라벨은 본 서비스가 부여(`SPEAKER_00`, ...). 사람 이름 매핑은 **speech_recognize 서비스** 책임.

### 5.3 Alignment
- 사전 로드 언어: `["en", "ko"]`. 필요 언어는 lifespan 코드에 추가.
- 가중치 누락 시 warning 후 계속 진행(해당 언어만 align 비활성).

---

## 6. Job 관리

- 현재는 **메모리 dict (`jobs`)** 기반. 단일 프로세스/단일 워커 가정.
- 다중 워커(`uvicorn --workers > 1`) 사용 금지. 필요 시 Redis 기반 잡 저장소로 이행 PR.
- Job 결과는 영구 저장하지 않음. 백엔드는 **완료 즉시 결과를 pull 하여 자체 DB에 저장** 해야 한다.

---

## 7. Docker / 배포 표준

- Base image: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`.
- `ENV WHISPER_MODEL_DIR=/app/src/resources/models`, `APP_PORT=6002`.
- 모델은 이미지에 포함하지 않고 볼륨 마운트.
- `.dockerignore`: `src/resources/models/`, `src/test/*.wav`, `.venv/`, `nohup.out`, `server.log`, `__pycache__/`.

### 7.1 Jenkins 연동
- `IMG_STT_API=<registry>/stt-api:<env>`.
- compose 서비스명: `stt_api`. 내부 URL: `STT_API_BASE=http://stt_api:8000`.
- 포트: dev 6002 / stg 9002 / prd 8002.

---

## 8. 로깅 / 관측

| 항목 | 표준 |
|------|------|
| 요청 로그 | `[Job {id}] Processing {filename}` |
| 진행 로그 | `step` / `percent` 단위 |
| 완료 로그 | `Completed in Ns` |
| 실패 로그 | `logger.error(..., exc_info=True)` + `progress=failed_at_*` |
| `/health` | 항상 200. payload 로 모델 상태 표시 |

---

## 9. 보안 / 데이터

- 업로드 음성 파일은 임시 디렉토리에 저장 후, 백그라운드 작업 종료 시점에 **반드시 삭제** (`background_process` 의 finally 절 유지).
- 응답에 포함되는 텍스트는 민감정보일 수 있음 → 백엔드는 로그/저장 정책에 맞게 처리.
- `HF_TOKEN` 등 시크릿은 코드/이미지에 포함하지 않고 환경변수로만 전달.

---

## 10. 백엔드 연동 시 주의

- **항상 비동기 Job 패턴**. 동기 응답을 기대하지 말 것.
- 큰 파일(>30분)은 GPU 1장당 동시 1~2건만 권장. 백엔드 큐로 직렬화 권장.
- 결과는 단명(in-memory). **백엔드가 폴링 완료 즉시 가져가서 자체 저장.**
- 화자 라벨은 익명(`SPEAKER_00`). 사람 매핑은 **speech_recognize** 호출 단계에서 수행.
- CORS는 와일드카드로 열려있음 — 운영에선 ingress(nginx) 단에서 제한.

---

## 11. Git / 브랜치 / PR

- 브랜치: `feat/stt-<topic>`, `fix/stt-<topic>`.
- 커밋: `[stt] <동사> <내용>`.
- 모델/오디오 파일 커밋 금지(`*.bin`, `*.wav`, `*.pt`, `*.onnx`).
- WhisperX git pin 변경 시 PR 본문에 **벤치(언어별 WER, 처리 속도)** 첨부 권장.
