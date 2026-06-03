PYTHON ?= .venv/bin/python
CONFIG ?= config.yaml

.PHONY: check compile test run experiment

check: compile test run experiment

compile:
	$(PYTHON) -m compileall main.py experiment.py scripts src

test:
	$(PYTHON) -m unittest discover -s tests

run:
	$(PYTHON) main.py --config $(CONFIG)

experiment:
	$(PYTHON) experiment.py --config $(CONFIG) --max-images 2