.PHONY: help build gui gui-build gui-test phase19-report phase19-verify reward-components-freeze reward-components-dry-run reward-components-run reward-components-verify cyborg-build cyborg-dirs cyborg-smoke cyborg-train cyborg-eval cyborg-verify cyberbattle-build cyberbattle-dirs cyberbattle-smoke cyberbattle-train cyberbattle-eval cyberbattle-verify phase14-report phase14-verify phase15-report phase15-verify recurrent-freeze recurrent-dry-run recurrent-dev recurrent-run recurrent-verify curriculum-freeze curriculum-dry-run curriculum-dev curriculum-run curriculum-verify graph-freeze graph-dry-run graph-dev graph-run graph-verify transfer-freeze transfer-dry-run transfer-dev transfer-run transfer-verify hierarchical-freeze hierarchical-dry-run hierarchical-dev hierarchical-run hierarchical-verify multiagent-freeze multiagent-dry-run multiagent-dev multiagent-run multiagent-verify onprem-train onprem-eval onprem-verify infrastructure-train infrastructure-eval hybrid-smoke hybrid-train hybrid-eval lab-build lab-plan lab-scan test test-fast test-slow test-one lint db-up db-down db-summary db-shell rollout enterprise-demo onprem-demo train train-sparse experiment-freeze experiment-dry-run experiment catalogue manifest verify-nvd clean

.PHONY: neo4j-build neo4j-up neo4j-down neo4j-export neo4j-analyze neo4j-verify
.PHONY: mlflow-build mlflow-up mlflow-down mlflow-sync mlflow-verify
.PHONY: convergence-dirs convergence-1m-freeze convergence-1m-run convergence-1m-analyze convergence-normalized-freeze convergence-normalized-run convergence-normalized-analyze convergence-verify convergence-sparse-run convergence-evaluate reproducibility-check

# Historical evidence is read-only. Only the NEW convergence namespace is writable.
CONVERGENCE_CONFIG ?= configs/experiments/experiment_01_convergence_1m_v2.yaml
CONVERGENCE_NORMALIZED := configs/experiments/experiment_01_convergence_normalized_1m_v2.yaml
CONVERGENCE_RUN = podman run --rm --network sourcecode_rlredteam-internal \
	--cap-drop=all --security-opt=no-new-privileges --user 10001:10001 \
	--memory=12g --pids-limit=512 \
	--tmpfs /app/runs:rw,mode=1777 \
	-e PYTHONHASHSEED=0 -e MPLCONFIGDIR=/tmp/matplotlib \
	-e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
	-e RLREDTEAM_GIT_DIRTY="$(RLREDTEAM_GIT_DIRTY)" \
	-e RLREDTEAM_IMAGE_DIGEST="$$(podman image inspect localhost/sourcecode_app:latest --format '{{.Id}}')" \
	-e POSTGRES_HOST=postgres -e POSTGRES_PORT=5432 \
	-e POSTGRES_USER -e POSTGRES_PASSWORD -e POSTGRES_DB \
	-v "$(CURDIR):/app:ro,z" \
	-v "$(CURDIR)/results/convergence_v2:/app/results/convergence_v2:rw,z" \
	-w /app localhost/sourcecode_app:latest

convergence-dirs: ## Prepare only the new convergence evidence directory
	podman run --rm --network none --user 10001:10001 \
		-v "$(CURDIR)/results:/app/results:rw,z" localhost/sourcecode_app:latest \
		python -c 'from pathlib import Path; Path("/app/results/convergence_v2").mkdir(exist_ok=True)'

convergence-1m-freeze: convergence-dirs ## Preregister inputs/criterion (NOT a convergence claim)
	set -a; . ./.env; set +a; $(CONVERGENCE_RUN) python scripts/run_convergence.py freeze --config $(CONVERGENCE_CONFIG)

convergence-1m-run: convergence-dirs db-up ## Explicit LONG run: ten 1M-step shaped seeds
	set -a; . ./.env; set +a; $(CONVERGENCE_RUN) python scripts/run_convergence.py run --config $(CONVERGENCE_CONFIG)

convergence-1m-analyze: convergence-dirs ## Reconstruct formal assessment; freeze first passing config
	set -a; . ./.env; set +a; $(CONVERGENCE_RUN) python scripts/run_convergence.py analyze --config $(CONVERGENCE_CONFIG)

convergence-normalized-freeze: convergence-dirs ## Only after baseline assessed and failed
	set -a; . ./.env; set +a; $(CONVERGENCE_RUN) python scripts/run_convergence.py freeze --config $(CONVERGENCE_NORMALIZED)

convergence-normalized-run: convergence-dirs db-up ## Explicit LONG reward-normalization comparison
	set -a; . ./.env; set +a; $(CONVERGENCE_RUN) python scripts/run_convergence.py run --config $(CONVERGENCE_NORMALIZED)

convergence-normalized-analyze: convergence-dirs ## Assess normalization and compare raw reward curves
	set -a; . ./.env; set +a; $(CONVERGENCE_RUN) python scripts/run_convergence.py analyze --config $(CONVERGENCE_NORMALIZED)

convergence-verify: convergence-dirs ## Fast protocol tests; never launch 1M training
	set -a; . ./.env; set +a; $(CONVERGENCE_RUN) pytest -q -p no:cacheprovider tests/test_convergence.py tests/test_convergence_runner.py
	set -a; . ./.env; set +a; $(CONVERGENCE_RUN) python scripts/run_convergence.py verify --config $(CONVERGENCE_CONFIG)

convergence-sparse-run: convergence-dirs db-up ## Gated matched sparse training, after shaped freeze only
	set -a; . ./.env; set +a; $(CONVERGENCE_RUN) python scripts/run_convergence.py sparse-run --config $(CONVERGENCE_CONFIG)

convergence-evaluate: convergence-dirs db-up ## Gated frozen-policy n=10 paired evaluation
	set -a; . ./.env; set +a; $(CONVERGENCE_RUN) python scripts/run_convergence.py evaluate --config $(CONVERGENCE_CONFIG)

reproducibility-check: convergence-dirs db-up ## Compute topology SHA and execute Module 0/4 tests, no fixed counts
	set -a; . ./.env; set +a; $(CONVERGENCE_RUN) python scripts/check_reproducibility.py

export UID := $(shell id -u)
export GID := $(shell id -g)
# The container has only a partial checkout, so dirtiness is decided here.
export RLREDTEAM_GIT_DIRTY := $(shell test -n "$$(git status --porcelain 2>/dev/null)" && echo 1 || echo 0)

# Podman is the sole supported container runtime for this project.
COMPOSE := podman compose
NEO4J_COMPOSE := podman compose -f docker-compose.yml -f docker-compose.neo4j.yml
NEO4J_CLIENT_IMAGE := localhost/security-rl-neo4j-client:latest
NEO4J_RUN := podman run --rm --network sourcecode_rlredteam-internal \
	--cap-drop=all --security-opt=no-new-privileges --user 10001:10001 \
	-e PYTHONHASHSEED=0 -e POSTGRES_HOST=postgres -e POSTGRES_PORT=5432 \
	-e POSTGRES_USER -e POSTGRES_PASSWORD -e POSTGRES_DB \
	-e NEO4J_URI=bolt://neo4j:7687 -e NEO4J_USER -e NEO4J_PASSWORD \
	-e NEO4J_DATABASE -v "$(CURDIR)/src:/app/src:ro,z" \
	-v "$(CURDIR)/tests:/app/tests:ro,z" \
	-v "$(CURDIR)/scripts:/app/scripts:ro,z" \
	-v "$(CURDIR)/configs:/app/configs:ro,z" \
	-v "$(CURDIR)/data:/app/data:ro,z" \
	-v "$(CURDIR)/.git:/app/.git:ro,z" \
	-v "$(CURDIR)/docker-compose.yml:/app/docker-compose.yml:ro,z" \
	-v "$(CURDIR)/docker-compose.neo4j.yml:/app/docker-compose.neo4j.yml:ro,z" \
	-v "$(CURDIR)/Dockerfile.neo4j:/app/Dockerfile.neo4j:ro,z" \
	-v "$(CURDIR)/pyproject.toml:/app/pyproject.toml:ro,z" \
	-v "$(CURDIR)/results:/app/results:rw,z" -w /app $(NEO4J_CLIENT_IMAGE)
MLFLOW_COMPOSE := podman compose -f docker-compose.yml -f docker-compose.mlflow.yml
MLFLOW_IMAGE := localhost/security-rl-mlflow:latest
MLFLOW_RUN := podman run --rm --network sourcecode_rlredteam-internal \
	--cap-drop=all --security-opt=no-new-privileges --user 10001:10001 \
	-e PYTHONHASHSEED=0 -e POSTGRES_HOST=postgres -e POSTGRES_PORT=5432 \
	-e POSTGRES_USER -e POSTGRES_PASSWORD -e POSTGRES_DB \
	-e MLFLOW_TRACKING_URI=http://mlflow:5000 -e MLFLOW_EXPERIMENT_PREFIX \
	-e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
	-v "$(CURDIR)/src:/app/src:ro,z" \
	-v "$(CURDIR)/tests:/app/tests:ro,z" \
	-v "$(CURDIR)/scripts:/app/scripts:ro,z" \
	-v "$(CURDIR)/configs:/app/configs:ro,z" \
	-v "$(CURDIR)/data:/app/data:ro,z" \
	-v "$(CURDIR)/.git:/app/.git:ro,z" \
	-v "$(CURDIR)/docker-compose.yml:/app/docker-compose.yml:ro,z" \
	-v "$(CURDIR)/docker-compose.mlflow.yml:/app/docker-compose.mlflow.yml:ro,z" \
	-v "$(CURDIR)/Dockerfile.mlflow:/app/Dockerfile.mlflow:ro,z" \
	-v "$(CURDIR)/pyproject.toml:/app/pyproject.toml:ro,z" \
	-v "$(CURDIR)/runs:/app/runs:ro,z" \
	-v "$(CURDIR)/results:/app/results:rw,z" -w /app $(MLFLOW_IMAGE)
CYBERBATTLE_IMAGE := localhost/security-rl-cyberbattle:latest
CYBERBATTLE_RUN := podman run --rm --network none --cap-drop=all \
	--security-opt=no-new-privileges --user 0:0 \
	-e PYTHONHASHSEED=0 -e MPLCONFIGDIR=/tmp/matplotlib \
	-e RLREDTEAM_GIT_DIRTY="$(RLREDTEAM_GIT_DIRTY)" \
	-v "$(CURDIR)/src:/app/src:ro,z" \
	-v "$(CURDIR)/tests:/app/tests:ro,z" \
	-v "$(CURDIR)/scripts:/app/scripts:ro,z" \
	-v "$(CURDIR)/configs:/app/configs:ro,z" \
	-v "$(CURDIR)/data:/app/data:ro,z" \
	-v "$(CURDIR)/.git:/app/.git:ro,z" \
	-v "$(CURDIR)/pyproject.toml:/app/pyproject.toml:ro,z" \
	-v "$(CURDIR)/runs:/app/runs:rw,z" \
	-v "$(CURDIR)/results:/app/results:rw,z" \
	-w /app $(CYBERBATTLE_IMAGE)
CYBORG_IMAGE := localhost/security-rl-cyborg:latest
CYBORG_RUN := podman run --rm --network none --cap-drop=all \
	--security-opt=no-new-privileges --user 0:0 \
	-e PYTHONHASHSEED=0 -e MPLCONFIGDIR=/tmp/matplotlib \
	-e RLREDTEAM_GIT_DIRTY="$(RLREDTEAM_GIT_DIRTY)" \
	-v "$(CURDIR)/src:/app/src:ro,z" \
	-v "$(CURDIR)/tests:/app/tests:ro,z" \
	-v "$(CURDIR)/scripts:/app/scripts:ro,z" \
	-v "$(CURDIR)/configs:/app/configs:ro,z" \
	-v "$(CURDIR)/data:/app/data:ro,z" \
	-v "$(CURDIR)/.git:/app/.git:ro,z" \
	-v "$(CURDIR)/pyproject.toml:/app/pyproject.toml:ro,z" \
	-v "$(CURDIR)/runs:/app/runs:rw,z" \
	-v "$(CURDIR)/results:/app/results:rw,z" \
	-w /app $(CYBORG_IMAGE)

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

reward-components-freeze: ## Preregister the Phase 18 component-level reward study
	podman run --rm --user 0 \
		-v "$$PWD:/app:rw,z" -w /app localhost/sourcecode_app:latest \
		python scripts/run_reward_component_ablation.py --freeze

reward-components-dry-run: ## Validate Phase 18 hashes and print its 21-run grid
	$(COMPOSE) run --rm app python scripts/run_reward_component_ablation.py --dry-run

reward-components-run: ## Train/evaluate all matched Phase 18 reward arms
	$(COMPOSE) up -d postgres
	$(COMPOSE) run --rm app python scripts/run_reward_component_ablation.py

reward-components-verify: ## Verify Phase 18 policies, pairing and frozen inputs
	$(COMPOSE) run --rm app pytest -q -p no:cacheprovider \
		tests/test_reward.py tests/test_component_ablation.py
	$(COMPOSE) run --rm app python scripts/verify_reward_component_ablation.py

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

cyberbattle-build: ## Build isolated CyberBattleSim image without changing NASim dependencies
	podman image exists localhost/sourcecode_app:latest || $(COMPOSE) build app
	podman build -t $(CYBERBATTLE_IMAGE) -f Dockerfile.cyberbattle .

cyberbattle-dirs: cyberbattle-build ## Create only the dedicated writable evidence directories
	podman run --rm --network none --user 0:0 \
		-v "$(CURDIR)/runs:/app/runs:rw,z" \
		-v "$(CURDIR)/results:/app/results:rw,z" $(CYBERBATTLE_IMAGE) \
		sh -c 'mkdir -p /app/runs/cyberbattle /app/results/cyberbattle'

cyberbattle-smoke: cyberbattle-dirs ## Run scripted toy-chain trajectory and Phase 14 report
	$(CYBERBATTLE_RUN) python scripts/cyberbattle_smoke.py

cyberbattle-train: cyberbattle-dirs ## Train standard PPO on the small CyberBattle chain
	$(CYBERBATTLE_RUN) python scripts/train_cyberbattle.py

cyberbattle-eval: cyberbattle-dirs ## Deterministically evaluate frozen PPO on held-out chain/seeds
	$(CYBERBATTLE_RUN) python scripts/evaluate_cyberbattle.py

cyberbattle-verify: cyberbattle-build ## Test and verify CyberBattle artifacts and frozen NASim isolation
	$(CYBERBATTLE_RUN) pytest -q -p no:cacheprovider tests/test_simulator_adapter.py tests/test_cyberbattle_adapter.py
	$(CYBERBATTLE_RUN) python scripts/verify_cyberbattle_completion.py

cyborg-build: ## Build the pinned, dependency-isolated CybORG image
	podman image exists localhost/sourcecode_app:latest || $(COMPOSE) build app
	podman build -t $(CYBORG_IMAGE) -f Dockerfile.cyborg .

cyborg-dirs: cyborg-build ## Create only the dedicated CybORG evidence directories
	podman run --rm --network none --user 0:0 \
		-v "$(CURDIR)/runs:/app/runs:rw,z" \
		-v "$(CURDIR)/results:/app/results:rw,z" $(CYBORG_IMAGE) \
		sh -c 'mkdir -p /app/runs/cyborg /app/results/cyborg'

cyborg-smoke: cyborg-dirs ## Run the observable-only Scenario1 path and Phase 14 report
	$(CYBORG_RUN) python scripts/cyborg_smoke.py

cyborg-train: cyborg-dirs ## Train standard single-agent PPO through the CybORG adapter
	$(CYBORG_RUN) python scripts/train_cyborg.py

cyborg-eval: cyborg-dirs ## Deterministically evaluate the frozen CybORG PPO policy
	$(CYBORG_RUN) python scripts/evaluate_cyborg.py

cyborg-verify: cyborg-build ## Test and verify CybORG evidence and hidden-state isolation
	$(CYBORG_RUN) pytest -q -p no:cacheprovider tests/test_simulator_adapter.py tests/test_cyborg_adapter.py
	$(CYBORG_RUN) python scripts/verify_cyborg_completion.py

phase14-report: ## Generate and persist deterministic explainable attack-path report
	$(COMPOSE) up -d postgres
	$(COMPOSE) run --rm app python scripts/generate_attack_path_report.py --postgres

phase14-verify: ## Test and reconstruct Phase 14 report from source + PostgreSQL
	$(COMPOSE) up -d postgres
	$(COMPOSE) run --rm app python scripts/verify_phase14_completion.py --postgres

phase15-report: ## Paired frozen-policy evaluation with one path-critical CVE disabled
	$(COMPOSE) up -d postgres
	$(COMPOSE) run --rm app python scripts/run_mitigation_counterfactual.py \
		--run runs/experiment_01-shaped-s42-t42 \
		--seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 \
		--cve CVE-2024-6387 \
		--phase14-report results/phase14/experiment_01_attack_path_report.json \
		--protect results/experiment_01 \
		--out results/phase15/experiment_01_mitigation.json --postgres

phase15-verify: ## Verify Phase 15 pairing, policy/artifact immutability and PostgreSQL
	$(COMPOSE) up -d postgres
	$(COMPOSE) run --rm app python scripts/verify_phase15_completion.py --postgres

phase19-report: ## Generate/persist the evidence-only causal path and knowledge graph
	$(COMPOSE) up -d postgres
	$(COMPOSE) run --rm app python scripts/generate_causal_attack_graph.py --postgres

phase19-verify: ## Verify Phase 19 causality, UI behavior and PostgreSQL reconstruction
	$(COMPOSE) up -d postgres
	$(COMPOSE) run --rm app python scripts/verify_causal_attack_graph.py --postgres
	podman run --rm -e QT_QPA_PLATFORM=offscreen \
		-v "$(CURDIR):/app:ro,z" -w /app rlredteam-gui \
		python -m pytest -q -p no:cacheprovider tests/test_gui_causal_graph.py

neo4j-build: ## Build the pinned Neo4j-client application image
	podman image exists localhost/sourcecode_app:latest || $(COMPOSE) build app
	podman build -t $(NEO4J_CLIENT_IMAGE) -f Dockerfile.neo4j .

neo4j-up: neo4j-build ## Start the optional loopback-only derived graph service
	$(NEO4J_COMPOSE) up -d postgres neo4j

neo4j-down: ## Stop Phase 20 services without deleting graph/database volumes
	$(NEO4J_COMPOSE) down

neo4j-export: neo4j-up ## Rebuild a selected Neo4j projection from PostgreSQL
	set -a; . ./.env; set +a; $(NEO4J_RUN) python scripts/export_neo4j_projection.py

neo4j-analyze: neo4j-up ## Run graph-scoped derived analysis queries
	set -a; . ./.env; set +a; $(NEO4J_RUN) python scripts/analyze_neo4j_projection.py

neo4j-verify: neo4j-up ## Verify source integrity, projection and analysis end to end
	set -a; . ./.env; set +a; $(NEO4J_RUN) pytest -q -p no:cacheprovider \
		tests/test_neo4j_projection.py tests/test_neo4j_security.py \
		tests/test_neo4j_integration.py
	set -a; . ./.env; set +a; $(NEO4J_RUN) python scripts/verify_neo4j_completion.py

mlflow-build: ## Build the pinned, dependency-isolated MLflow image
	podman image exists localhost/sourcecode_app:latest || $(COMPOSE) build app
	podman build -t $(MLFLOW_IMAGE) -f Dockerfile.mlflow .

mlflow-up: mlflow-build ## Start optional loopback-only MLflow observability
	$(MLFLOW_COMPOSE) up -d postgres mlflow

mlflow-down: ## Stop Phase 21 services without deleting mirror/database volumes
	$(MLFLOW_COMPOSE) down

mlflow-sync: mlflow-up ## Mirror recent PostgreSQL runs into MLflow
	set -a; . ./.env; set +a; $(MLFLOW_RUN) python scripts/sync_mlflow_observability.py

mlflow-verify: mlflow-up ## Verify idempotence and authoritative-source immutability
	set -a; . ./.env; set +a; $(MLFLOW_RUN) pytest -q -p no:cacheprovider \
		tests/test_mlflow_observability.py tests/test_mlflow_security.py \
		tests/test_mlflow_integration.py
	set -a; . ./.env; set +a; $(MLFLOW_RUN) python scripts/verify_mlflow_completion.py

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

onprem-train: ## Train mask-aware PPO across on-prem topology seeds 1-60
	$(COMPOSE) run --rm app python scripts/train_onprem.py

onprem-eval: ## Evaluate frozen on-prem PPO on held-out seeds 2001-2020
	$(COMPOSE) run --rm app python scripts/evaluate_onprem.py --split test --postgres

onprem-verify: ## Verify gated on-prem artifacts and reconstruct evaluations from PostgreSQL
	$(COMPOSE) up -d postgres
	$(COMPOSE) run --rm app python scripts/verify_onprem_completion.py --postgres

infrastructure-train: ## Train one mask-aware PPO across legacy/cloud/hybrid profiles
	$(COMPOSE) run --rm app python scripts/train_infrastructure.py

infrastructure-eval: ## Evaluate frozen PPO on all held-out infrastructure profiles
	$(COMPOSE) run --rm app python scripts/evaluate_infrastructure.py --split test --postgres

recurrent-freeze: ## Freeze Phase 8 recurrent-study source/config hashes
	podman run --rm --user 0:0 -w /app -e MPLCONFIGDIR=/tmp/matplotlib \
		-e RLREDTEAM_GIT_DIRTY="$$RLREDTEAM_GIT_DIRTY" \
		-v "$$PWD/configs:/app/configs:rw,z" \
		-v "$$PWD/src:/app/src:ro,z" \
		-v "$$PWD/scripts:/app/scripts:ro,z" \
		-v "$$PWD/.git:/app/.git:ro,z" \
		-v "$$PWD/pyproject.toml:/app/pyproject.toml:ro,z" \
		localhost/sourcecode_app:latest python scripts/run_recurrent_study.py freeze

recurrent-dry-run: ## Validate Phase 8 frozen inputs and list the 20-run grid
	$(COMPOSE) run --rm app python scripts/run_recurrent_study.py dry-run

recurrent-dev: ## Run excluded-seed Phase 8 feasibility and validation study
	$(COMPOSE) run --rm app python scripts/run_recurrent_study.py development

recurrent-run: ## Run canonical matched Phase 8 training/test evaluation in PostgreSQL
	$(COMPOSE) run --rm app python scripts/run_recurrent_study.py run --postgres

recurrent-verify: ## Verify Phase 8 files, checkpoints and PostgreSQL reconstruction
	$(COMPOSE) run --rm app python scripts/verify_recurrent_completion.py --postgres

curriculum-freeze: ## Freeze Phase 9 curriculum-study source/config hashes
	podman run --rm --user 0:0 -w /app -e MPLCONFIGDIR=/tmp/matplotlib \
		-e RLREDTEAM_GIT_DIRTY="$$RLREDTEAM_GIT_DIRTY" \
		-v "$$PWD/configs:/app/configs:rw,z" \
		-v "$$PWD/src:/app/src:ro,z" \
		-v "$$PWD/scripts:/app/scripts:ro,z" \
		-v "$$PWD/.git:/app/.git:ro,z" \
		-v "$$PWD/pyproject.toml:/app/pyproject.toml:ro,z" \
		localhost/sourcecode_app:latest python scripts/run_curriculum_study.py freeze

curriculum-dry-run: ## Validate Phase 9 frozen inputs and list the 20-run grid
	$(COMPOSE) run --rm app python scripts/run_curriculum_study.py dry-run

curriculum-dev: ## Run excluded-seed Phase 9 feasibility and validation study
	$(COMPOSE) run --rm app python scripts/run_curriculum_study.py development

curriculum-run: ## Run canonical matched Phase 9 training/test evaluation in PostgreSQL
	$(COMPOSE) run --rm app python scripts/run_curriculum_study.py run --postgres

curriculum-verify: ## Verify Phase 9 files, checkpoints and PostgreSQL reconstruction
	$(COMPOSE) run --rm app python scripts/verify_curriculum_completion.py --postgres

graph-freeze: ## Freeze Phase 10 graph-policy source/config hashes
	podman run --rm --user 0:0 -w /app -e MPLCONFIGDIR=/tmp/matplotlib \
		-e RLREDTEAM_GIT_DIRTY="$$RLREDTEAM_GIT_DIRTY" \
		-v "$$PWD/configs:/app/configs:rw,z" \
		-v "$$PWD/src:/app/src:ro,z" \
		-v "$$PWD/scripts:/app/scripts:ro,z" \
		-v "$$PWD/.git:/app/.git:ro,z" \
		-v "$$PWD/pyproject.toml:/app/pyproject.toml:ro,z" \
		localhost/sourcecode_app:latest python scripts/run_graph_policy_study.py freeze

graph-dry-run: ## Validate Phase 10 frozen inputs and list the 20-run grid
	$(COMPOSE) run --rm app python scripts/run_graph_policy_study.py dry-run

graph-dev: ## Run excluded-seed Phase 10 feasibility and validation study
	$(COMPOSE) run --rm app python scripts/run_graph_policy_study.py development

graph-run: ## Run canonical matched Phase 10 training/test evaluation in PostgreSQL
	$(COMPOSE) run --rm app python scripts/run_graph_policy_study.py run --postgres

graph-verify: ## Verify Phase 10 files, checkpoints and PostgreSQL reconstruction
	$(COMPOSE) run --rm app python scripts/verify_graph_policy_completion.py --postgres

transfer-freeze: ## Freeze Phase 11 transfer-learning source/config hashes
	podman run --rm --user 0:0 -w /app -e MPLCONFIGDIR=/tmp/matplotlib \
		-e RLREDTEAM_GIT_DIRTY="$$RLREDTEAM_GIT_DIRTY" \
		-v "$$PWD/configs:/app/configs:rw,z" \
		-v "$$PWD/src:/app/src:ro,z" \
		-v "$$PWD/scripts:/app/scripts:ro,z" \
		-v "$$PWD/.git:/app/.git:ro,z" \
		-v "$$PWD/pyproject.toml:/app/pyproject.toml:ro,z" \
		localhost/sourcecode_app:latest python scripts/run_transfer_study.py freeze

transfer-dry-run: ## Validate Phase 11 frozen inputs and list the 20-run grid
	$(COMPOSE) run --rm app python scripts/run_transfer_study.py dry-run

transfer-dev: ## Run excluded-seed Phase 11 feasibility and validation study
	$(COMPOSE) run --rm app python scripts/run_transfer_study.py development

transfer-run: ## Run canonical matched Phase 11 training/test evaluation in PostgreSQL
	$(COMPOSE) run --rm app python scripts/run_transfer_study.py run --postgres

transfer-verify: ## Verify Phase 11 files, checkpoints and PostgreSQL reconstruction
	$(COMPOSE) run --rm app python scripts/verify_transfer_completion.py --postgres

hierarchical-freeze: ## Freeze Phase 12 hierarchical-policy source/config hashes
	podman run --rm --user 0:0 -w /app -e MPLCONFIGDIR=/tmp/matplotlib \
		-e RLREDTEAM_GIT_DIRTY="$$RLREDTEAM_GIT_DIRTY" \
		-v "$$PWD/configs:/app/configs:rw,z" \
		-v "$$PWD/src:/app/src:ro,z" \
		-v "$$PWD/scripts:/app/scripts:ro,z" \
		-v "$$PWD/.git:/app/.git:ro,z" \
		-v "$$PWD/pyproject.toml:/app/pyproject.toml:ro,z" \
		localhost/sourcecode_app:latest python scripts/run_hierarchical_study.py freeze

hierarchical-dry-run: ## Validate Phase 12 frozen inputs and list the 20-run grid
	$(COMPOSE) run --rm app python scripts/run_hierarchical_study.py dry-run

hierarchical-dev: ## Run excluded-seed Phase 12 feasibility and validation study
	$(COMPOSE) run --rm app python scripts/run_hierarchical_study.py development

hierarchical-run: ## Run canonical matched Phase 12 training/test evaluation in PostgreSQL
	$(COMPOSE) run --rm app python scripts/run_hierarchical_study.py run --postgres

hierarchical-verify: ## Verify Phase 12 files, checkpoints and PostgreSQL reconstruction
	$(COMPOSE) run --rm app python scripts/verify_hierarchical_completion.py --postgres

multiagent-freeze: ## Freeze Phase 13 red-blue study source/config/defender hashes
	podman run --rm --user 0:0 -w /app -e MPLCONFIGDIR=/tmp/matplotlib \
		-e RLREDTEAM_GIT_DIRTY="$$RLREDTEAM_GIT_DIRTY" \
		-v "$$PWD/configs:/app/configs:rw,z" \
		-v "$$PWD/src:/app/src:ro,z" \
		-v "$$PWD/scripts:/app/scripts:ro,z" \
		-v "$$PWD/tests:/app/tests:ro,z" \
		-v "$$PWD/runs:/app/runs:ro,z" \
		-v "$$PWD/results:/app/results:ro,z" \
		-v "$$PWD/docs:/app/docs:ro,z" \
		-v "$$PWD/.git:/app/.git:ro,z" \
		-v "$$PWD/pyproject.toml:/app/pyproject.toml:ro,z" \
		localhost/sourcecode_app:latest python scripts/run_multiagent_study.py freeze

multiagent-dry-run: ## Validate Phase 13 frozen inputs and list the 20-run grid
	$(COMPOSE) run --rm app python scripts/run_multiagent_study.py dry-run

multiagent-dev: ## Run full-budget excluded-seed Phase 13 red-blue development gate
	$(COMPOSE) run --rm app python scripts/run_multiagent_study.py development

multiagent-run: ## Run canonical Phase 13 matched study with PostgreSQL telemetry
	$(COMPOSE) run --rm app python scripts/run_multiagent_study.py run --postgres

multiagent-verify: ## Verify Phase 13 artifacts and exact PostgreSQL reconstruction
	$(COMPOSE) run --rm app python scripts/verify_multiagent_completion.py --postgres

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
