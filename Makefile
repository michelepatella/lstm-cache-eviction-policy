# ----------------------------
# Dependencies
# ----------------------------

# Install dependencies
install:
	pip install -r requirements.txt

# Update requirements.txt
update-deps:
	pip freeze > requirements.txt

# ----------------------------
# Code Quality
# ----------------------------

# Formatting
format:
	black src

# (Imports) Sorting
sort-imports:
	isort src

# Linting
lint:
	flake8 src

# Typing
type-check:
	mypy src

# ----------------------------
# Documentation
# ----------------------------

# Doc generation with pdoc
.PHONY: docs

docs:
	PYTHONPATH=$(shell pwd)/src pdoc --html src/ --output-dir docs/_build/html --force

# ----------------------------
# Cleanup
# ----------------------------

# Clean logs
clean-logs:
	rm -rf logs/*/*.log
	rm -rf logs/*/*.log.*

# Clean pycache
clean-pyc:
	find . -name "__pycache__" -exec rm -rf {} +
