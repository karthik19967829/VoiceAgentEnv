FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Demo UI + pre-recorded showcase + minimal audio assets
COPY voiceenv /app/voiceenv
COPY hvb_samples/audio/agent/00d676d7058c49bb.wav /app/hvb_samples/audio/agent/00d676d7058c49bb.wav
COPY hvb_samples/transcript/00d676d7058c49bb.json /app/hvb_samples/transcript/00d676d7058c49bb.json
COPY environments/auto_00d676d7 /app/environments/auto_00d676d7

ENV VOICEENV_SHOWCASE=1
ENV PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/api/mode || exit 1

CMD ["uvicorn", "voiceenv.ui.demo_app:app", "--host", "0.0.0.0", "--port", "7860"]
