# Atalhos de desenvolvimento (Linux, macOS ou WSL). No Windows sem make, use os
# comandos equivalentes do README.

.DEFAULT_GOAL := help
.PHONY: help env up down reset logs dev smoke test test-pg lint fmt

help: ## lista os comandos
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-9s %s\n", $$1, $$2}'

env: ## cria o .env (com uma SWITCHBOARD_SECRET_KEY aleatória)
	@if [ -f .env ]; then echo ".env já existe"; else \
		key=$$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))'); \
		sed "s|^SWITCHBOARD_SECRET_KEY=.*|SWITCHBOARD_SECRET_KEY=$$key|" .env.example > .env; \
		echo ".env criado"; fi

up: env ## sobe postgres, console, router e agentes de exemplo (Docker)
	docker compose up -d --build
	@echo "console: http://localhost:8000  ·  router: http://localhost:8080/docs"

down: ## derruba a stack (mantém o banco)
	docker compose down

reset: ## derruba a stack e apaga o banco
	docker compose down -v

logs: ## acompanha os logs
	docker compose logs -f --tail=100

dev: ## roda tudo local sem Docker (SQLite em ./data)
	./scripts/dev.sh

smoke: ## testa a stack no ar de ponta a ponta (RAG, tool MCP, agentes A2A, needs_input)
	python3 scripts/smoke.py

test: ## testes (SQLite)
	uv run pytest -q

test-pg: ## testes incluindo PostgreSQL (precisa de um pgvector em localhost:5432)
	SWITCHBOARD_TEST_DATABASE_URL=$${SWITCHBOARD_TEST_DATABASE_URL:-postgresql://switchboard:switchboard@localhost:5432/switchboard_test} uv run pytest -q

lint: ## ruff (lint + formatação)
	uv run ruff check . && uv run ruff format --check .

fmt: ## corrige lint e formata
	uv run ruff check --fix . && uv run ruff format .
