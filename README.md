# WhisperX On-Premise API

이 프로젝트는 인터넷 연결이 제한된 환경(온프레미스)에서 WhisperX 모델을 API 서비스로 제공하기 위한 구성을 담고 있습니다.
`whisperX` 소스 코드(`../whisperX`)를 직접 참조하여 동작합니다.

## 1. 사전 요구사항 (Prerequisites)

- Python 3.10+
- NVIDIA GPU (권장, CUDA 드라이버 설치 필요).
- ffmpeg (오디오 처리를 위해 필수)
- uv (패키지 매니저)

## 2. 설치 (Setup)

`uv`를 사용하여 환경을 구성합니다. `pyproject.toml`에 상위 폴더의 `whisperX`가 로컬 의존성으로 등록되어 있습니다.

```bash
# ffmpeg 설치
sudo apt-get update && sudo apt-get install ffmpeg

# uv 설치 (미설치 시)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 의존성 설치 (whisperX 포함)
uv sync
```

## 3. 모델 다운로드 (Model Download)

인터넷이 가능한 환경에서 모델을 다운로드합니다 (`large-v3` 기본).

```bash
uv run src/resources/download_models.py --output ./models
```

## 4. API 서버 실행 (Run Server)

서버를 실행합니다. 기본 포트는 `8012`입니다.

```bash
uv run src/v1/main.py
```

또는 개발 모드(코드 변경 시 자동 재시작):
```bash
uv run uvicorn src.v1.main:app --host 0.0.0.0 --port 8012 --reload
```

## 5. API 사용 예시

```bash
curl -X POST "http://localhost:8012/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/audio.mp3" \
  -F "language=ko" \
  -F "align=true"
```
