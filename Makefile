.PHONY: install install-dev run test docker-up docker-down

install: ## Install runtime dependencies
	pip install -r requirements.txt

install-dev: ## Install development/test dependencies
	pip install -r requirements.txt -r requirements-dev.txt

run: ## Start the FastAPI server
	python main.py

test: ## Run the test suite
	pytest

docker-up: ## Build and start with Docker Compose
	docker compose up --build

docker-down: ## Stop Docker Compose services
	docker compose down
