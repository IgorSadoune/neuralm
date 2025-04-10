# Makefile for neuralpy package

.PHONY: clean clean-test clean-pyc clean-build docs help install dev test lint dist build check upload

help:
	@python -c "$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

install: clean ## install the package to the active Python's site-packages
	pip install .

install-dev: clean ## install the package in development mode with dev dependencies
	pip install -e .
	pip install -r requirements.txt

lint: ## check style with flake8
	flake8 neuralm tests

test: ## run tests quickly with the default Python
	python -m unittest discover tests

test-all: ## run tests on every Python version with tox
	tox

build: clean ## builds source and wheel package
	python -m build

dist: build ## alias for build

check: ## check package for PyPI submission
	twine check dist/*

upload: build check ## upload package to PyPI
	twine upload dist/*

upload-test: build check ## upload package to Test PyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
