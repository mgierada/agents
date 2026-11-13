DATA ?= data/raw
OUT  ?= data/clean
# VLM  ?= qwen2.5vl:7b
VLM  ?= qwen3-vl:4b
SCOPE ?= borderline
# gallery audit
GALLERY ?= ../highfive-models/pre_processed/logo_audit_batches_1-10_17-31_with_pseudolabels__2024-11-22__distill-8baece-c0.7
TARGETS ?= ../highfive-models/highfive_models/data_processing/target_lists/target_logos_2024-11-25.txt
RUN ?= calib
# JUDGE ?= bedrock-converse:qwen.qwen3-vl-235b-a22b
JUDGE ?= bedrock-converse:global.anthropic.claude-haiku-5-5
# JUDGE ?= ollama:qwen2.5vl:7b
# JUDGE ?= openai:gpt-6-astra
# JUDGE ?= ollama:qwen3-vl:4b
# JUDGE ?= bedrock:claude-haiku-5-5
# JUDGE ?= claude:claude-haiku-5-5
# JUDGE ?= claude:claude-opus-5-5
GARGS = --gallery-dir $(GALLERY) --target-list $(TARGETS)

.DEFAULT_GOAL := help
.PHONY: help install run dry-run run-vlm ollama-serve ollama-pull graph stats lint format typecheck test check clean clean-output \
	gallery-calib gallery-review gallery-resolve gallery-calibrate gallery-plot gallery-ui

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Create venv and install deps (incl. dev)
	uv sync

run: ## Run cleanup: DATA -> OUT
	uv run cleanup run --data-dir $(DATA) --output-dir $(OUT)

dry-run: ## Assess DATA and write report only
	uv run cleanup run --data-dir $(DATA) --output-dir $(OUT) --dry-run

run-vlm: ## Run cleanup with a VLM judge (VLM=claude:claude-opus-5-5|ollama:qwen2.5vl:7b, SCOPE=...)
	uv run cleanup run --data-dir $(DATA) --output-dir $(OUT) --judge-model $(VLM) --judge-scope $(SCOPE)

gallery-calib: ## Gallery: new run RUN with a 300-crop calibration sample
	uv run cleanup gallery init --run $(RUN) $(GARGS) --calibration-sample 300

gallery-review: ## Gallery: VLM review of RUN (JUDGE=..., ARGS="--max-units 20")
	uv run cleanup gallery review --run $(RUN) $(GARGS) --judge $(JUDGE) $(ARGS)

gallery-resolve: ## Gallery: decisions + before/after for RUN
	uv run cleanup gallery resolve --run $(RUN) $(GARGS) $(ARGS)

gallery-calibrate: ## Gallery: judge/flag precision vs your labels in RUN
	uv run cleanup gallery calibrate --run $(RUN) $(GARGS)

gallery-plot: ## Gallery: calibration charts (PNG) into RUN/plots/
	uv run cleanup gallery plot --run $(RUN) $(GARGS)

gallery-ui: ## Gallery: label/review UI for RUN on http://127.0.0.1:8765
	uv run cleanup gallery ui --run $(RUN) $(GARGS)

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

lint-fix: ##@ attempt to fix all fixable linting issues
	uv run ruff check --fix

typecheck: ## ty type check
	uv run ty check

test: ## Run pytest
	uv run pytest

check: lint typecheck test ## Lint, typecheck, test

clean: ## Remove caches
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +

clean-output: ## Delete OUT directory
	rm -rf $(OUT)
