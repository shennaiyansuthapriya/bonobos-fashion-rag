.PHONY: install dev up down migrate seed test lint

install:
	pip install -r requirements.txt

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8004

up:
	docker compose -f docker/docker-compose.yml up -d

down:
	docker compose -f docker/docker-compose.yml down

migrate:
	alembic upgrade head

seed:
	python scripts/seed_sample_data.py

test:
	pytest tests/ -v --tb=short

lint:
	ruff check app/ tests/

format:
	ruff format app/ tests/
