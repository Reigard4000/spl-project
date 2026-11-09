API_HOST ?= 127.0.0.1
API_PORT ?= 8000

.PHONY: install install-geracl data train retrain evaluate run-api run-ui run-client metrics test lint clean clean-models

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

install-geracl:
	python -m pip install -r requirements-geracl.txt

data:
	python data/generate_data.py

train: data
	python scripts/train_detectors.py

retrain: clean-models train

evaluate: data
	python scripts/evaluate.py

run-api:
	uvicorn src.presentation.api.app:create_app --factory --host $(API_HOST) --port $(API_PORT) --reload

run-ui:
	PYTHONPATH=. streamlit run src/presentation/ui/streamlit_app.py

run-client:
	python client/server.py

metrics: data
	python scripts/compute_metrics.py

test:
	pytest

lint:
	ruff check .
	mypy src scripts data tests client

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -f models/pr_curve.png data/detector_train.csv data/clean_texts.txt data/anomaly_texts.txt spl_logs.db

clean-models:
	rm -f models/*.pkl models/metrics.json models/pr_curve.png
