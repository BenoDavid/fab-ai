# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (gcc often helps catboost wheels; also locales)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src /app/src

# Create default dirs (can be mounted over)
RUN mkdir -p /app/model /app/data

EXPOSE 8000

# Default command (can be overridden)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]