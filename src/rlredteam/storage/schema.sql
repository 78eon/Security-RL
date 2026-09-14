-- Module 4: episode logging schema.
-- Loaded automatically by docker-entrypoint-initdb.d on first container start.

CREATE TABLE IF NOT EXISTS experiments (
    id                   BIGSERIAL PRIMARY KEY,
    name                 TEXT        NOT NULL,
    reward_mode          TEXT        NOT NULL,
    -- Everything needed to prove two runs were comparable. The ablation is only
    -- valid if the arms share topology_config_hash and cve_manifest_sha256 and
    -- differ solely in reward_mode.
    config_hash          TEXT        NOT NULL,
    topology_config_hash TEXT        NOT NULL,
    cve_manifest_sha256  TEXT        NOT NULL,
    git_sha              TEXT,
    seed_set             INTEGER[]   NOT NULL,
    notes                TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Additive research-provenance migration for databases created by an older
-- image. CREATE TABLE IF NOT EXISTS does not add columns to an existing table.
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS condition TEXT;
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS algorithm TEXT NOT NULL DEFAULT 'PPO';
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS topology_id TEXT;
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS topology_hash TEXT;
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS hyperparameters JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS runs (
    id             BIGSERIAL PRIMARY KEY,
    experiment_id  BIGINT      NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    seed           INTEGER     NOT NULL,
    designation    TEXT        NOT NULL CHECK (designation IN ('training', 'evaluation')),
    status         TEXT        NOT NULL CHECK (status IN ('running', 'complete', 'failed')),
    evaluation_seeds INTEGER[] NOT NULL DEFAULT '{}',
    checkpoint_path TEXT,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at       TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS episodes (
    id                  BIGSERIAL PRIMARY KEY,
    experiment_id       BIGINT      NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    seed                INTEGER     NOT NULL,
    topology_seed       INTEGER     NOT NULL,
    episode_idx         INTEGER     NOT NULL,
    total_reward        DOUBLE PRECISION NOT NULL,
    -- Always recorded, in every reward mode, so results can be reported in
    -- native NASim units. Comparing raw shaped return across arms is invalid --
    -- each arm would be measured with its own ruler.
    native_reward       DOUBLE PRECISION NOT NULL,
    length              INTEGER     NOT NULL,
    terminal_state      TEXT        NOT NULL,
    goal_reached        BOOLEAN     NOT NULL,
    exploited_hosts     JSONB       NOT NULL DEFAULT '[]'::jsonb,
    mean_cvss_exploited DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (experiment_id, seed, episode_idx)
);

ALTER TABLE episodes ADD COLUMN IF NOT EXISTS run_id BIGINT REFERENCES runs(id) ON DELETE CASCADE;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS max_cvss_exploited DOUBLE PRECISION;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS hosts_compromised INTEGER;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS topology_hash TEXT;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS known_nodes INTEGER;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS true_nodes INTEGER;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS discovery_coverage DOUBLE PRECISION;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS invalid_mask_selections INTEGER NOT NULL DEFAULT 0;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS failed_actions INTEGER NOT NULL DEFAULT 0;
ALTER TABLE episodes ADD COLUMN IF NOT EXISTS deployment_profile TEXT;

CREATE TABLE IF NOT EXISTS steps (
    id            BIGSERIAL PRIMARY KEY,
    episode_id    BIGINT  NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    step_idx      INTEGER NOT NULL,
    action_name   TEXT    NOT NULL,
    action_kind   TEXT    NOT NULL,
    rl_action_index INTEGER CONSTRAINT steps_rl_action_index_nonnegative
        CHECK (rl_action_index IS NULL OR rl_action_index >= 0),
    simulator_action TEXT NOT NULL,
    framework_mappings JSONB NOT NULL DEFAULT '[]'::jsonb,
    tactic        TEXT,
    technique_id  TEXT,
    target_subnet INTEGER,
    target_host   INTEGER,
    success       BOOLEAN NOT NULL,
    reward        DOUBLE PRECISION NOT NULL,
    native_reward DOUBLE PRECISION NOT NULL,
    cve_id        TEXT,
    cvss_base     DOUBLE PRECISION
);

ALTER TABLE steps ADD COLUMN IF NOT EXISTS cve_term DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS tactic_term DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS discovery_term DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS crown_jewel_term DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS penalty_term DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS access_gained INTEGER NOT NULL DEFAULT 0;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS newly_discovered INTEGER NOT NULL DEFAULT 0;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS is_crown_jewel BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS reward_paid BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS error TEXT;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS target_entity TEXT;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS state_changed BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS prerequisites JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS outcomes JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS rl_action_index INTEGER;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS simulator_action TEXT;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS framework_mappings JSONB NOT NULL DEFAULT '[]'::jsonb;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'steps_rl_action_index_nonnegative'
    ) THEN
        ALTER TABLE steps ADD CONSTRAINT steps_rl_action_index_nonnegative
            CHECK (rl_action_index IS NULL OR rl_action_index >= 0);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_episodes_experiment_seed ON episodes (experiment_id, seed);
CREATE INDEX IF NOT EXISTS idx_steps_episode ON steps (episode_id);
CREATE INDEX IF NOT EXISTS idx_steps_cve ON steps (cve_id) WHERE cve_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_steps_framework_mappings
    ON steps USING GIN (framework_mappings)
    WHERE framework_mappings <> '[]'::jsonb;

CREATE OR REPLACE VIEW step_framework_techniques AS
SELECT
    steps.id AS step_id,
    steps.episode_id,
    steps.step_idx,
    steps.rl_action_index,
    COALESCE(steps.simulator_action, steps.action_name) AS simulator_action,
    mapping ->> 'framework' AS framework,
    mapping ->> 'mapping_version' AS mapping_version,
    mapping ->> 'tactic_id' AS tactic_id,
    mapping ->> 'tactic_name' AS tactic_name,
    mapping ->> 'technique_id' AS technique_id,
    mapping ->> 'technique_name' AS technique_name,
    mapping ->> 'source_url' AS source_url
FROM steps
CROSS JOIN LATERAL jsonb_array_elements(steps.framework_mappings) AS mapping;

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs (experiment_id);
CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes (run_id) WHERE run_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_run_idx
    ON episodes (run_id, episode_idx) WHERE run_id IS NOT NULL;

-- Phase 14 derived reporting. Raw episodes/steps remain immutable inputs; the
-- GUI reads this persisted deterministic payload instead of recomputing facts.
CREATE TABLE IF NOT EXISTS attack_path_reports (
    report_id                    TEXT PRIMARY KEY,
    report_mode                  TEXT NOT NULL
        CHECK (report_mode IN ('simulation_report', 'evidence_report')),
    experiment_key               TEXT NOT NULL,
    run_keys                     TEXT[] NOT NULL,
    source_trajectory_sha256     TEXT NOT NULL CHECK (length(source_trajectory_sha256) = 64),
    facts_payload_sha256         TEXT NOT NULL CHECK (length(facts_payload_sha256) = 64),
    mitre_catalogue_sha256       TEXT NOT NULL CHECK (length(mitre_catalogue_sha256) = 64),
    mitre_catalogue_version      TEXT NOT NULL,
    source_checkpoint_hashes     JSONB NOT NULL DEFAULT '{}'::jsonb,
    report_schema_version        TEXT NOT NULL,
    report_data                  JSONB NOT NULL,
    created_at                   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS attack_path_report_facts (
    report_id   TEXT NOT NULL REFERENCES attack_path_reports(report_id) ON DELETE CASCADE,
    fact_index  INTEGER NOT NULL CHECK (fact_index >= 0),
    episode_key TEXT NOT NULL,
    fact_data   JSONB NOT NULL,
    PRIMARY KEY (report_id, fact_index)
);

CREATE TABLE IF NOT EXISTS attack_path_report_criticalities (
    report_id        TEXT NOT NULL REFERENCES attack_path_reports(report_id) ON DELETE CASCADE,
    cve_id           TEXT NOT NULL,
    criticality_data JSONB NOT NULL,
    PRIMARY KEY (report_id, cve_id)
);

CREATE INDEX IF NOT EXISTS idx_attack_path_reports_experiment
    ON attack_path_reports (experiment_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_attack_path_report_facts_episode
    ON attack_path_report_facts (report_id, episode_key);

-- Phase 15 derived counterfactual evaluation. These tables never update the
-- Phase 14 report or the raw training/evaluation episode tables.
CREATE TABLE IF NOT EXISTS mitigation_counterfactual_reports (
    report_id                  TEXT PRIMARY KEY,
    selected_cve               TEXT NOT NULL,
    intervention_version       TEXT NOT NULL,
    intervention_sha256        TEXT NOT NULL CHECK (length(intervention_sha256) = 64),
    checkpoint_sha256          TEXT NOT NULL CHECK (length(checkpoint_sha256) = 64),
    source_attack_report_id    TEXT,
    evaluation_seeds           INTEGER[] NOT NULL,
    report_schema_version      TEXT NOT NULL,
    report_data                JSONB NOT NULL,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mitigation_counterfactual_pairs (
    report_id       TEXT NOT NULL
        REFERENCES mitigation_counterfactual_reports(report_id) ON DELETE CASCADE,
    evaluation_seed INTEGER NOT NULL,
    original_data   JSONB NOT NULL,
    mitigated_data  JSONB NOT NULL,
    delta_data      JSONB NOT NULL,
    PRIMARY KEY (report_id, evaluation_seed)
);

CREATE INDEX IF NOT EXISTS idx_mitigation_reports_cve
    ON mitigation_counterfactual_reports (selected_cve, created_at DESC);

-- Phase 19 evidence-backed derived graph. PostgreSQL is authoritative for the
-- graph projection; every node/edge points back to immutable knowledge evidence.
CREATE TABLE IF NOT EXISTS causal_attack_graphs (
    graph_id        TEXT PRIMARY KEY CHECK (length(graph_id) = 64),
    report_id       TEXT NOT NULL,
    schema_version  TEXT NOT NULL,
    provenance      JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS causal_knowledge_evidence (
    graph_id             TEXT NOT NULL REFERENCES causal_attack_graphs(graph_id) ON DELETE CASCADE,
    evidence_id          TEXT NOT NULL CHECK (length(evidence_id) = 64),
    episode_key          TEXT NOT NULL,
    step_idx             INTEGER NOT NULL CHECK (step_idx >= 0),
    source_event_sha256  TEXT NOT NULL,
    evidence_kind        TEXT NOT NULL,
    evidence_data        JSONB NOT NULL,
    PRIMARY KEY (graph_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS causal_node_facts (
    graph_id              TEXT NOT NULL REFERENCES causal_attack_graphs(graph_id) ON DELETE CASCADE,
    node_id               TEXT NOT NULL,
    node_type             TEXT NOT NULL,
    first_discovery_step  TEXT NOT NULL,
    discovered_by         TEXT NOT NULL,
    evidence_ids          TEXT[] NOT NULL,
    node_data             JSONB NOT NULL,
    PRIMARY KEY (graph_id, node_id)
);

CREATE TABLE IF NOT EXISTS causal_edge_facts (
    graph_id           TEXT NOT NULL REFERENCES causal_attack_graphs(graph_id) ON DELETE CASCADE,
    edge_id            TEXT NOT NULL CHECK (length(edge_id) = 64),
    source_node        TEXT NOT NULL,
    target_node        TEXT NOT NULL,
    relationship_type  TEXT NOT NULL CHECK (relationship_type IN (
        'DISCOVERED', 'REVEALED', 'YIELDED_CREDENTIAL', 'GRANTS_ACCESS',
        'EXPLOITED', 'AUTHENTICATED_TO', 'PIVOTED_TO', 'CONNECTED_TO',
        'HOSTS', 'EXPOSES', 'CONTAINS'
    )),
    episode_key        TEXT NOT NULL,
    step_idx           INTEGER NOT NULL CHECK (step_idx >= 0),
    evidence_ids       TEXT[] NOT NULL CHECK (cardinality(evidence_ids) > 0),
    edge_data          JSONB NOT NULL,
    PRIMARY KEY (graph_id, edge_id)
);

CREATE INDEX IF NOT EXISTS idx_causal_graph_report
    ON causal_attack_graphs (report_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_causal_edge_source_target
    ON causal_edge_facts (graph_id, source_node, target_node);
CREATE INDEX IF NOT EXISTS idx_causal_evidence_step
    ON causal_knowledge_evidence (graph_id, episode_key, step_idx);
