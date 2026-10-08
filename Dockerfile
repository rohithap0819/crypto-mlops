FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System packages needed by scientific Python / ML libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY src ./src

# Runtime directories are supplied through volumes.
RUN mkdir -p \
    /app/data \
    /app/models \
    /app/reports

ENV PYTHONPATH=/app

EXPOSE 8000
EXPOSE 8501

CMD ["python", "--version"]