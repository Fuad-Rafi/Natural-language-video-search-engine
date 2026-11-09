# VideoSeek — CPU demo image.
# For GPU, use Dockerfile.gpu. Model weights are downloaded on first run.

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIDEOSEEK_HOST=0.0.0.0 \
    VIDEOSEEK_PORT=7860 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/frames data/indexes data/audio data/transcripts data/ocr data/captions logs

EXPOSE 7860
CMD ["python", "app.py"]
