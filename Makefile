PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv install dev test eval svg smoke seed clean

help:
	@echo "make install  创建虚拟环境并安装依赖"
	@echo "make dev      启动服务 http://127.0.0.1:8000"
	@echo "make test     pytest"
	@echo "make eval     跑评测集，输出 14 项指标"
	@echo "make smoke    端到端冒烟（需先启动服务）"
	@echo "make seed     语料体检并重建索引"
	@echo "make svg      重新生成架构图"

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install -q -r requirements.txt

dev:
	$(PY) -m uvicorn legalcourt.api.main:app --app-dir src --reload --host 127.0.0.1 --port 8000

test:
	PYTHONPATH=src $(PY) -m pytest -q

eval:
	PYTHONPATH=src $(PY) -m legalcourt.eval.run_eval

smoke:
	PYTHONPATH=src $(PY) scripts/smoke_api.py

seed:
	PYTHONPATH=src $(PY) scripts/seed_corpus.py --reindex

svg:
	$(PY) scripts/make_architecture_svg.py

clean:
	rm -rf data/runtime data/index .pytest_cache
