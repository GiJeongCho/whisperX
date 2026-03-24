# WhisperX On-Premise API

이 프로젝트는 인터넷 연결이 제한된 환경(온프레미스)에서 WhisperX 모델을 API 서비스로 제공하기 위한 구성을 담고 있습니다.
`whisperX` 소스 코드(`../whisperX`)를 직접 참조하여 동작합니다.

## 1. 사전 요구사항 (Prerequisites)

- Docker + Docker Compose
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- 모델 파일 (아래 "모델 다운로드" 섹션 참고)

## 2. 환경 설정 (Configuration)

```bash
cp .env.example .env
```

`.env` 파일을 필요에 맞게 수정합니다:

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `STT_PORT` | `8000` | 호스트에 노출할 포트 |
| `MODEL_DIR` | `./models` | 모델 파일 경로 (호스트) |
| `WHISPER_ARCH` | `large-v3` | Whisper 모델 아키텍처 |
| `HF_TOKEN` | _(빈 값)_ | Hugging Face 토큰 (Pyannote 모델 다운로드용) |

## 3. 모델 다운로드 (Model Download)

인터넷이 가능한 환경에서 모델을 다운로드합니다 (`large-v3` 기본).

```bash
python src/resources/download_models.py --output ./models
```

## 4. Docker 빌드 및 실행

```bash
# 빌드 및 실행
docker compose up -d --build

# 로그 확인
docker compose logs -f stt

# 중지
docker compose down
```

이미지만 빌드하려면:

```bash
docker build -t stt-server .
```

## 5. API 사용 예시

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/audio.mp3" \
  -F "language=ko" \
  -F "align=true"
```

헬스체크:

```bash
curl http://localhost:8000/health
```

## 6. 로컬 개발 (선택)

Docker 없이 로컬에서 직접 실행하려면:

```bash
# uv 설치 (미설치 시)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# 서버 실행
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
