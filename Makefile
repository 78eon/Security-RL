.PHONY: help build gui gui-build gui-test hybrid-smoke hybrid-train hybrid-eval lab-build lab-plan lab-scan test test-fast test-slow test-one lint db-up db-down db-summary db-shell rollout enterprise-demo onprem-demo train train-sparse experiment-freeze experiment-dry-run experiment catalogue manifest verify-nvd clean

export UID := $(shell id -u)
export GID := $(shell id -g)
# The container has only a partial checkout, so dirtiness is decided here.
export RLREDTEAM_GIT_DIRTY := $(shell test -n "$$(git status --porcelain 2>/dev/null)" && echo 1 || echo 0)

# Podman is the sole supported container runtime for this project.
COMPOSE := podman compose

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
	$(COMPOSE) run --rm app ruff check --cache-dir /tmp/ruff src gui tests tools scripts

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

experiment-freeze: ## Preregister hashes for the canonical fixed-topology experiment
	podman run --rm --user 0 \
		-v "$$PWD/configs:/app/configs:rw,z" \
		-v "$$PWD/data:/app/data:ro,z" \
		-v "$$PWD/.git:/app/.git:ro,z" \
		localhost/sourcecode_app:latest python scripts/run_experiment.py \
		--config configs/experiments/experiment_01.yaml --freeze

experiment-dry-run: ## Validate frozen inputs and print the 20-run grid
	$(COMPOSE) run --rm app python scripts/run_experiment.py \
		--config configs/experiments/experiment_01.yaml --dry-run

experiment:     ## Train, evaluate and package the canonical Essential experiment
	$(COMPOSE) up -d postgres
	$(COMPOSE) run --rm app python scripts/run_experiment.py \
		--config configs/experiments/experiment_01.yaml

gui-build:      ## Build the desktop GUI image (separate from training)
	podman build -t rlredteam-gui -f Dockerfile.gui .

gui:             ## Launch the native containerized research console
	@test -f .env || { echo "no .env — copy .env.example and set credentials"; exit 1; }
	xhost +local: >/dev/null 2>&1 || true
	set -a; . ./.env; set +a; \
	podman run --rm -e DISPLAY="$$DISPLAY" \
		-e POSTGRES_USER -e POSTGRES_PASSWORD -e POSTGRES_DB \
		-e POSTGRES_HOST=127.0.0.1 -e POSTGRES_PORT=5433 \
		-e RLREDTEAM_HOST_REPO="$$PWD" \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v "$$PWD:/app:ro,z" -w /app --net=host rlredteam-gui python -m gui

gui-test:       ## Headless tests for the desktop GUI and adapter
	podman run --rm -e QT_QPA_PLATFORM=offscreen \
		-v "$$PWD:/app:ro,z" -w /app rlredteam-gui \
		python -m pytest tests/test_gui*.py -q -p no:cacheprovider

lab-build:      ## Build the unprivileged isolated-range discovery image
	podman build -t rlredteam-lab -f Dockerfile.lab .

lab-plan:       ## Dry-run a scoped scan: make lab-plan T=10.250.0.10 P=host_discovery
	@test -n "$(T)" || { echo "set T to an authorized private IP/CIDR"; exit 1; }
	podman run --rm --cap-drop=all --security-opt=no-new-privileges \
		-v "$$PWD:/app:ro,z" -w /app rlredteam-lab \
		python scripts/lab_discover.py --config configs/lab_scope.yaml \
		--target "$(T)" --profile "$(or $(P),host_discovery)"

lab-scan:       ## Authorized live scan; set T, P and A (authorization ID)
	@test -n "$(T)" -a -n "$(A)" || { echo "set T and authorization A"; exit 1; }
	podman run --rm --network host --cap-drop=all --security-opt=no-new-privileges \
		-v "$$PWD:/app:ro,z" -w /app rlredteam-lab \
		python scripts/lab_discover.py --config configs/lab_scope.yaml \
		--target "$(T)" --profile "$(or $(P),host_discovery)" \
		--execute --authorization "$(A)"

rollout:        ## Deterministic random-policy rollout on the frozen topology
	$(COMPOSE) run --rm app python scripts/rollout_random.py --seed 42

enterprise-demo: ## Typed enterprise discovery and attack-path demonstration
	$(COMPOSE) run --rm app python scripts/enterprise_demo.py --seed 42

onprem-demo: ## Hidden seeded on-prem discovery and attack-path feasibility demo
	$(COMPOSE) run --rm app python scripts/onprem_demo.py --seed 2001

hybrid-smoke:   ## Feasibility baseline on three held-out hybrid topologies
	$(COMPOSE) run --rm app python scripts/evaluate_hybrid.py --split test --limit 3

hybrid-train:   ## Train PPO across hybrid simulation seeds 1-60
	$(COMPOSE) run --rm app python scripts/train_hybrid.py --seed 42 --timesteps 50000

hybrid-eval:    ## Evaluate frozen hybrid PPO on held-out seeds 2001-2020
	$(COMPOSE) run --rm app python scripts/evaluate_hybrid.py \
		--model runs/hybrid-ppo/model-seed-42.zip --split test

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
