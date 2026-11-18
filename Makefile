.PHONY: help install test lint coverage healthcheck docker-build docker-up docker-down docker-logs clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	pip install -r requirements.txt

test:  ## Run test suite with pytest
	pytest -v

lint:  ## Run code verification and syntax compilation checks
	python -m py_compile main.py config.py diagnostics.py logging_config.py db/*.py io/*.py pipeline/*.py tests/*.py

coverage:  ## Run pytest test suite with code coverage reporting
	coverage run -m pytest -v
	coverage report -m

healthcheck:  ## Run local system and dependency health check
	python -m main --check-health

docker-build:  ## Build segmentation pipeline Docker image
	docker build -t medseg-pipeline:latest .

docker-up:  ## Start PostgreSQL database and pipeline services with healthcheck
	docker compose up -d

docker-down:  ## Stop all Docker Compose services and networks
	docker compose down

docker-logs:  ## View streaming logs from all Docker Compose services
	docker compose logs -f

clean:  ## Clean temporary Python caches and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
