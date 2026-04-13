FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV WHISPER_MODEL_DIR=/app/src/resources/models
ENV APP_PORT=6002

EXPOSE ${APP_PORT}
CMD ["sh", "-c", "python -m uvicorn src.api:app --host 0.0.0.0 --port ${APP_PORT}"]
