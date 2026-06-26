-- Meta-read pipeline as chained SQL views over the indexed replay_field table.
-- replay_field(episode_id, player, archetype, deck_sig, team)  [built by build_field_table.py]
-- replays(episode_id, team0, team1, reward0, reward1, source, blob)  [reward = winner signal]
--
-- Apply:  sqlite3 replays.sqlite < tools/field_views.sql
-- Then every meta question is a SELECT, e.g.:
--   SELECT * FROM v_archetype_field   ORDER BY appearances DESC;
--   SELECT * FROM v_archetype_winrate ORDER BY winrate_pct DESC;
--   SELECT * FROM v_matchup WHERE arch_a='Dragapult' ORDER BY games DESC;

-- 1) BASE: one row per episode = both players' archetype/deck + the game result.
DROP VIEW IF EXISTS v_episode_arch;
CREATE VIEW v_episode_arch AS
SELECT f0.episode_id,
       f0.archetype AS arch0, f1.archetype AS arch1,
       f0.deck_sig  AS sig0,  f1.deck_sig  AS sig1,
       r.reward0, r.reward1
FROM replay_field f0
JOIN replay_field f1 ON f1.episode_id = f0.episode_id AND f1.player = 1
JOIN replays       r ON r.episode_id  = f0.episode_id
WHERE f0.player = 0;

-- 2) FIELD: how common each archetype is (appearances + how many distinct lists).
DROP VIEW IF EXISTS v_archetype_field;
CREATE VIEW v_archetype_field AS
SELECT archetype,
       COUNT(*)                 AS appearances,
       COUNT(DISTINCT deck_sig) AS distinct_decks
FROM replay_field
GROUP BY archetype;

-- 3) DECKS: every distinct 60-card list, its archetype, frequency, a sample team name.
DROP VIEW IF EXISTS v_top_decks;
CREATE VIEW v_top_decks AS
SELECT deck_sig, archetype, COUNT(*) AS appearances, MAX(team) AS sample_team
FROM replay_field
GROUP BY deck_sig;

-- 4) WIN-RATE: per-archetype overall win rate (each game contributes 2 player-rows).
DROP VIEW IF EXISTS v_archetype_winrate;
CREATE VIEW v_archetype_winrate AS
SELECT arch AS archetype,
       COUNT(*)                              AS games,
       SUM(win)                              AS wins,
       ROUND(100.0 * SUM(win) / COUNT(*), 1) AS winrate_pct
FROM (
  SELECT arch0 AS arch, CASE WHEN reward0 > reward1 THEN 1 ELSE 0 END AS win FROM v_episode_arch
  UNION ALL
  SELECT arch1 AS arch, CASE WHEN reward1 > reward0 THEN 1 ELSE 0 END AS win FROM v_episode_arch
)
GROUP BY arch;

-- 5) MATCHUP: directed A-vs-B win rate (the meta-counter map).
DROP VIEW IF EXISTS v_matchup;
CREATE VIEW v_matchup AS
SELECT a AS arch_a, b AS arch_b,
       COUNT(*)                               AS games,
       SUM(awin)                              AS a_wins,
       ROUND(100.0 * SUM(awin) / COUNT(*), 1) AS a_winrate_pct
FROM (
  SELECT arch0 AS a, arch1 AS b, CASE WHEN reward0 > reward1 THEN 1 ELSE 0 END AS awin FROM v_episode_arch
  UNION ALL
  SELECT arch1 AS a, arch0 AS b, CASE WHEN reward1 > reward0 THEN 1 ELSE 0 END AS awin FROM v_episode_arch
)
GROUP BY a, b;
