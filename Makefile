DATA ?= data/raw
OUT  ?= data/clean
VLM  ?= qwen2.5vl:7b
SCOPE ?= borderline

.DEFAULT_GOAL := help
.PHONY: help install run dry-run run-vlm ollama-serve ollama-pull graph stats lint format typecheck test check clean clean-output

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Create venv and install deps (incl. dev)
	uv sync

run: ## Run cleanup: DATA -> OUT
	uv run cleanup run --data-dir $(DATA) --output-dir $(OUT)

dry-run: ## Assess DATA and write report only
	uv run cleanup run --data-dir $(DATA) --output-dir $(OUT) --dry-run

run-vlm: ## Run cleanup with Ollama VLM judge (VLM=..., SCOPE=borderline|survivors)
	uv run cleanup run --data-dir $(DATA) --output-dir $(OUT) --judge-model $(VLM) --judge-scope $(SCOPE)

ollama-serve: ## Start local Ollama server
	ollama serve

ollama-pull: ## Download the VLM model
	ollama pull $(VLM)

graph: ## Write pipeline graph as Mermaid to graph.mmd
	uv run cleanup graph > graph.mmd
	@echo "wrote graph.mmd"

stats: ## Count images/labels in DATA and OUT
	@for d in $(DATA) $(OUT); do \
		echo "$$d: images=$$(find $$d/images -type f 2>/dev/null | wc -l | tr -d ' ')" \
			"labels=$$(find $$d/labels -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')"; \
	done

lint: ## Ruff lint
	uv run ruff check .

format: ## Ruff format + autofix
	uv run ruff format .
	uv run ruff check . --fix

typecheck: ## ty type check
	uv run ty check

test: ## Run pytest
	uv run pytest

check: lint typecheck test ## Lint, typecheck, test

clean: ## Remove caches
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +

clean-output: ## Delete OUT directory
	rm -rf $(OUT)
