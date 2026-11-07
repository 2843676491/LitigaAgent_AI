#!/usr/bin/env bash
# 启动服务（本地单机形态）
set -e
cd "$(dirname "$0")/.."
. .venv/bin/activate 2>/dev/null || { echo "缺少 .venv，请先 make install"; exit 1; }
export PYTHONPATH=src
exec python -m uvicorn legalcourt.api.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
