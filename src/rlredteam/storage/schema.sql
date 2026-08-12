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

CREATE INDEX IF NOT EXISTS idx_episodes_experiment_seed ON episodes (experiment_id, seed);
CREATE INDEX IF NOT EXISTS idx_steps_episode ON steps (episode_id);
CREATE INDEX IF NOT EXISTS idx_steps_cve ON steps (cve_id) WHERE cve_id IS NOT NULL;
