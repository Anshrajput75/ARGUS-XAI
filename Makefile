.PHONY: install train test lint demo api dashboard experiments up
install:   ; pip install -e ".[all]"
train:     ; python -m argus.cli train
test:      ; pytest -q
lint:      ; ruff check src api tests experiments
api:       ; uvicorn api.main:app --reload
dashboard: ; streamlit run dashboard/app.py
experiments:
	for e in 01_baseline 02_gnn 03_unknown_attacks 04_xai 05_continual_learning; do \
	  python experiments/experiment_$$e/run.py --seeds 0 1 2 || exit 1; done
up:        ; docker compose -f docker/docker-compose.yml up --build
