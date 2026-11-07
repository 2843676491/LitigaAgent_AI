# 生产镜像（单机验证用；当前开发环境无 Docker，未实测）
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY web ./web
COPY data/corpus ./data/corpus
COPY data/samples ./data/samples

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "legalcourt.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
