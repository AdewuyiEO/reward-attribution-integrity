.PHONY: help setup synth run figures test dashboard clean all

help:
	@echo "make setup      install dependencies"
	@echo "make synth      generate synthetic data with planted fraud"
	@echo "make run        run the full detection pipeline"
	@echo "make figures    regenerate README figures"
	@echo "make test       run the test suite"
	@echo "make dashboard  launch the Streamlit monitor"
	@echo "make all        synth + run + figures + test"

setup:
	pip install -r requirements.txt

synth:
	python scripts/make_synthetic_data.py --rows 1200000

run:
	python -m src.pipeline

figures:
	python scripts/make_figures.py

test:
	pytest tests/ -q

dashboard:
	streamlit run dashboard/app.py

all: synth run figures test

clean:
	rm -rf outputs/*.parquet outputs/*.csv outputs/*.json data/*.duckdb
	find . -type d -name __pycache__ -exec rm -rf {} +
