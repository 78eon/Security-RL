.PHONY: help build test test-fast test-slow test-one lint db-up db-down db-summary db-shell rollout train train-sparse catalogue manifest verify-nvd clean

export UID := $(shell id -u)
export GID := $(shell id -g)

# This machine runs podman; CI or another machine may have docker. Auto-detect
# rather than hardcode, so `make test` works either way.
COMPOSE := $(shell command -v docker >/dev/null 2>&1 && echo "docker compose" || echo "podman compose")

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build:          ## Build the training image
	$(COMPOSE) build

db-up:          ## Start PostgreSQL (Module 4)
	$(COMPOSE) up -d postgres

db-down:        ## Stop PostgreSQL
	$(COMPOSE) down

db-summary:     ## Print what has been logged to PostgreSQL so far
	$(COMPOSE) run --rm app python -m rlredteam.storage.postgres_logger

db-shell:       ## Open a psql prompt against the project database
	$(COMPOSE) exec postgres psql -U rlredteam -d rlredteam

test:           ## Everything — 173 tests, ~90s. Run before every commit
	$(COMPOSE) up -d postgres
	$(COMPOSE) run --rm app pytest -q -p no:cacheprovider

test-fast:      ## Skip the training tests — ~6s. Use this while coding
	$(COMPOSE) run --rm app pytest -q -p no:cacheprovider -m "not slow"

test-slow:      ## Only the training tests, printing their measured numbers
	$(COMPOSE) up -d postgres
	$(COMPOSE) run --rm app pytest -p no:cacheprovider -m slow -v -s

test-one:       ## Run tests matching a name: make test-one T=farming
	$(COMPOSE) run --rm app pytest -p no:cacheprovider -k "$(T)" -v -s

lint:           ## Ruff
	$(COMPOSE) run --rm app ruff check src tests tools scripts

catalogue:      ## Rebuild the frozen SQLite CVE catalogue from data/provenance/
	$(COMPOSE) run --rm app python -m rlredteam.catalogue build

manifest:       ## Recompute and print the SHA-256 catalogue manifest
	$(COMPOSE) run --rm app python -m rlredteam.manifest

verify-nvd:     ## ONE-SHOT, ONLINE: diff the committed catalogue against live NVD
	python tools/fetch_nvd.py --verify

train:          ## 50k-step PPO pilot (shaped reward, seed 42)
	$(COMPOSE) run --rm app python -m rlredteam.train --seed 42 --timesteps 50000

train-sparse:   ## Same pilot on the sparse baseline
	$(COMPOSE) run --rm app python -m rlredteam.train --seed 42 --timesteps 50000 \
		--reward-config configs/sparse.yaml

rollout:        ## Deterministic random-policy rollout on the frozen topology
	$(COMPOSE) run --rm app python scripts/rollout_random.py --seed 42

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
