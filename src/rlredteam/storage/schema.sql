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

CREATE TABLE IF NOT EXISTS steps (
    id            BIGSERIAL PRIMARY KEY,
    episode_id    BIGINT  NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    step_idx      INTEGER NOT NULL,
    action_name   TEXT    NOT NULL,
    action_kind   TEXT    NOT NULL,
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
ALTER TABLE steps ADD COLUMN IF NOT EXISTS crown_jewel_term DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS penalty_term DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS access_gained INTEGER NOT NULL DEFAULT 0;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS newly_discovered INTEGER NOT NULL DEFAULT 0;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS is_crown_jewel BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS error TEXT;

CREATE INDEX IF NOT EXISTS idx_episodes_experiment_seed ON episodes (experiment_id, seed);
CREATE INDEX IF NOT EXISTS idx_steps_episode ON steps (episode_id);
CREATE INDEX IF NOT EXISTS idx_steps_cve ON steps (cve_id) WHERE cve_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs (experiment_id);
CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes (run_id) WHERE run_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_run_idx
    ON episodes (run_id, episode_idx) WHERE run_id IS NOT NULL;
