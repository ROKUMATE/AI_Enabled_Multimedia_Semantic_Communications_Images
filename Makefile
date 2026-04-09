PYTHON ?= .venv/bin/python
CONFIG ?= config.yaml

.PHONY: check compile run experiment

check: compile run experiment

compile:
	$(PYTHON) -m compileall main.py experiment.py src

run:
	$(PYTHON) main.py --config $(CONFIG) --no-enable-privacy

experiment:
	$(PYTHON) experiment.py --config $(CONFIG) --max-images 1 --noise-start 0.0 --noise-stop 0.5 --noise-step 0.5