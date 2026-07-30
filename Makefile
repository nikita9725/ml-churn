.PHONY: install sync run dev test lint format check docker-build docker-up docker-down

install:
	uv sync

sync:
	uv sync

run:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

dev:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

check: lint test

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
