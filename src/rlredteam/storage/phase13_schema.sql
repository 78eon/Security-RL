-- Phase 13 defensive telemetry extension.
-- Kept separate so earlier frozen study schemas remain byte-for-byte stable.

CREATE TABLE IF NOT EXISTS phase13_defense_episodes (
    episode_id              BIGINT PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
    native_total_reward     DOUBLE PRECISION NOT NULL,
    detected_actions        INTEGER NOT NULL CHECK (detected_actions >= 0),
    detectable_actions      INTEGER NOT NULL CHECK (detectable_actions > 0),
    detection_rate          DOUBLE PRECISION NOT NULL,
    evasion_rate            DOUBLE PRECISION NOT NULL,
    exploit_attempts        INTEGER NOT NULL CHECK (exploit_attempts >= 0),
    mitigated_exploits      INTEGER NOT NULL CHECK (mitigated_exploits >= 0),
    defender_actions        INTEGER NOT NULL CHECK (defender_actions >= 0),
    threshold_adjustments   INTEGER NOT NULL CHECK (threshold_adjustments >= 0),
    patch_schedules         INTEGER NOT NULL CHECK (patch_schedules >= 0),
    patch_activations       INTEGER NOT NULL CHECK (patch_activations >= 0)
);

CREATE TABLE IF NOT EXISTS phase13_defense_steps (
    step_id                     BIGINT PRIMARY KEY REFERENCES steps(id) ON DELETE CASCADE,
    native_reward               DOUBLE PRECISION NOT NULL,
    detected                    BOOLEAN NOT NULL,
    detection_probability       DOUBLE PRECISION NOT NULL,
    ids_level                   INTEGER NOT NULL CHECK (ids_level BETWEEN 0 AND 2),
    ids_level_name              TEXT NOT NULL,
    defender_action             TEXT NOT NULL,
    defender_action_index       INTEGER NOT NULL,
    defender_action_changed     BOOLEAN NOT NULL,
    scheduled_vulnerability     TEXT,
    patch_activation_episode    INTEGER,
    pending_patch_count         INTEGER NOT NULL CHECK (pending_patch_count >= 0),
    active_patch_count          INTEGER NOT NULL CHECK (active_patch_count >= 0),
    mitigation_blocked          BOOLEAN NOT NULL,
    defender_reward             DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_phase13_defense_episode
    ON phase13_defense_episodes (episode_id);
CREATE INDEX IF NOT EXISTS idx_phase13_defense_step
    ON phase13_defense_steps (step_id);
