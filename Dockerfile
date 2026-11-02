FROM python:3.12-slim

# OCR для сканованих PDF (українська + англійська) і утиліти для PDF
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ukr tesseract-ocr-eng poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot ./bot
COPY data ./data

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    ARCHIVE_DIR=/app/data/archive \
    TZ=Europe/Kyiv

VOLUME ["/data"]
CMD ["python", "-m", "bot.main"]
