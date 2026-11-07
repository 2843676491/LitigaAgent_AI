#!/usr/bin/env bash
# 跑评测集，产出 data/eval/results.json 与 docs/04-测试与评测报告.md
set -e
cd "$(dirname "$0")/.."
. .venv/bin/activate 2>/dev/null || { echo "缺少 .venv，请先 make install"; exit 1; }
export PYTHONPATH=src
exec python -m legalcourt.eval.run_eval
