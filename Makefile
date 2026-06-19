# Declare all phony targets
.PHONY: install format lint lint_tests clean test all

# Default target
.DEFAULT_GOAL := all

# Variables
SRC_PROJECT_NAME ?= src
SRC_TESTS ?= tests
FAIL_UNDER ?= 9.5
FAIL_UNDER_TESTS ?= 9.5
UV ?= uv
UVX ?= uvx

# Install project dependencies
install:
	@echo "Installing dependencies..."
	@$(UV) sync
	@echo "Dependencies installed! ✅"

# Code formatting
format:
	@echo "Formatting code..."
	@$(UVX) isort .
	@$(UVX) black .
	@echo "Formatting completed! ✅"

# Check code
lint:
	@echo "Running isort..."
	@$(UVX) isort $(SRC_PROJECT_NAME)
	@$(UVX) isort --check $(SRC_PROJECT_NAME)
	@echo "Running Black..."
	@$(UVX) black $(SRC_PROJECT_NAME)
	@$(UVX) black --check $(SRC_PROJECT_NAME)
	@echo "Running Bandit..."
	@$(UVX) bandit -r src
	@$(UVX) bandit -r tests --skip B101
	@echo "Running Mypy..."
	@$(UV) run mypy $(SRC_PROJECT_NAME)
	@echo "Running Flake8..."
	@$(UVX) flake8 --max-line-length=88 $(SRC_PROJECT_NAME)
	@echo "Running Ruff..."
	@$(UVX) ruff check $(SRC_PROJECT_NAME)
	@echo "Running Complexipy..."
	@$(UVX) complexipy $(SRC_PROJECT_NAME)
	@echo "Running Pylint..."
	@$(UV) run pylint --fail-under=$(FAIL_UNDER) $(SRC_PROJECT_NAME)
	@echo "Lint completed! ✅"

lint_tests:
	@echo "Running isort..."
	@$(UVX) isort $(SRC_TESTS)
	@$(UVX) isort --check $(SRC_TESTS)
	@echo "Running Black..."
	@$(UVX) black $(SRC_TESTS)
	@$(UVX) black --check $(SRC_TESTS)
	@echo "Running Bandit..."
	@$(UVX) bandit -r tests --skip B101
	@echo "Running Mypy..."
	@$(UV) run mypy $(SRC_TESTS)
	@echo "Running Flake8..."
	@$(UVX) flake8 --max-line-length=88 $(SRC_TESTS)
	@echo "Running Ruff..."
	@$(UVX) ruff check $(SRC_TESTS)
	@echo "Running Complexipy..."
	@$(UVX) complexipy $(SRC_TESTS)
	@echo "Running Pylint..."
	@$(UV) run pylint --fail-under=$(FAIL_UNDER_TESTS) $(SRC_TESTS)
	@echo "Lint completed! ✅"

# Clean cache and temporary files
clean:
	@echo "Cleaning cache and temporary files..."
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type d -name .pytest_cache -exec rm -rf {} +
	@find . -type d -name .mypy_cache -exec rm -rf {} +
	@find . -type d -name .ruff_cache -exec rm -rf {} +
	@find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	@rm -f .coverage
	@echo "Cleaning completed! ✅"

# Test the code (it is assumed tests are in the "tests" folder and start with "test_")
test:
	@echo "Checking if tests directory exists..."
	@if [ -d "$(SRC_TESTS)" ] && [ $$(find $(SRC_TESTS) -name "test_*.py" | wc -l) -gt 0 ]; then \
		echo "Running tests..."; \
		$(UV) run pytest tests; \
		echo "Tests passed! ✅"; \
	else \
		echo "No tests directory found or no test files. Skipping tests."; \
	fi

# Run all workflows
all: install format lint lint_tests test clean
	@echo "All tasks completed! ✅"
