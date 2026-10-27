FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/app/models

WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-deps .

COPY models/model.joblib models/metadata.json models/

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "readmission.api:app", "--host", "0.0.0.0", "--port", "8000"]
