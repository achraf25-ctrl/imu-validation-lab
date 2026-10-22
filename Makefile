.PHONY: install demo dashboard test all docker
install:
	pip install -e ".[dev]"
demo:            ## simulate all fault scenarios -> runs/
	python -m imulab demo --out runs
dashboard:       ## open the Streamlit dashboard on runs/
	streamlit run dashboard/app.py
test:
	python -m pytest
all: demo dashboard   ## the one-command run
docker:
	docker compose up --build
