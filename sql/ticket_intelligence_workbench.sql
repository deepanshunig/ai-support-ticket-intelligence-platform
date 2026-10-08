-- AI Support Ticket Intelligence Platform
-- MySQL Workbench setup and inspection script.
-- Safe to re-run: it does not drop tables or overwrite existing ticket rows.

CREATE DATABASE IF NOT EXISTS ticket_intelligence
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_0900_ai_ci;

USE ticket_intelligence;

CREATE TABLE IF NOT EXISTS tickets (
    id VARCHAR(36) NOT NULL,
    text TEXT NOT NULL,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    status VARCHAR(16) NOT NULL,
    version INT NOT NULL,
    prediction JSON NOT NULL,
    current_category VARCHAR(128) NOT NULL,
    assigned_team VARCHAR(128) NULL,
    model_version VARCHAR(256) NOT NULL,
    confidence DOUBLE NOT NULL,
    PRIMARY KEY (id),
    CONSTRAINT ticket_status
        CHECK (status IN ('needs_review', 'assigned', 'resolved')),
    CONSTRAINT ticket_positive_version CHECK (version >= 1),
    CONSTRAINT ticket_confidence_bounds
        CHECK (confidence >= 0 AND confidence <= 1),
    INDEX ix_ticket_status_created_at (status, created_at),
    INDEX ix_ticket_assigned_team_created_at (assigned_team, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS ticket_events (
    id INT NOT NULL AUTO_INCREMENT,
    ticket_id VARCHAR(36) NOT NULL,
    type VARCHAR(24) NOT NULL,
    created_at DATETIME(6) NOT NULL,
    actor_id VARCHAR(128) NOT NULL,
    payload JSON NOT NULL,
    PRIMARY KEY (id),
    INDEX ix_ticket_events_ticket_id (ticket_id),
    CONSTRAINT fk_ticket_events_ticket
        FOREIGN KEY (ticket_id) REFERENCES tickets (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS ticket_reviews (
    id INT NOT NULL AUTO_INCREMENT,
    ticket_id VARCHAR(36) NOT NULL,
    created_at DATETIME(6) NOT NULL,
    reviewer_id VARCHAR(128) NOT NULL,
    note TEXT NOT NULL,
    before_state JSON NOT NULL,
    after_state JSON NOT NULL,
    PRIMARY KEY (id),
    INDEX ix_ticket_reviews_ticket_id (ticket_id),
    CONSTRAINT fk_ticket_reviews_ticket
        FOREIGN KEY (ticket_id) REFERENCES tickets (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Quick checks in Workbench. Run after the setup statements above.
SHOW TABLES;

SELECT 'tickets' AS table_name, COUNT(*) AS row_count FROM tickets
UNION ALL
SELECT 'ticket_events', COUNT(*) FROM ticket_events
UNION ALL
SELECT 'ticket_reviews', COUNT(*) FROM ticket_reviews;

-- Recent tickets with their predicted category, confidence, and assigned team.
SELECT id, status, current_category, confidence, assigned_team, created_at
FROM tickets
ORDER BY created_at DESC
LIMIT 25;

-- Recent ticket audit history.
SELECT id, ticket_id, type, created_at, actor_id, payload
FROM ticket_events
ORDER BY created_at DESC
LIMIT 25;

-- Recent human reviews and corrections.
SELECT id, ticket_id, created_at, reviewer_id, note, before_state, after_state
FROM ticket_reviews
ORDER BY created_at DESC
LIMIT 25;
