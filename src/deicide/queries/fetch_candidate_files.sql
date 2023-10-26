WITH locs AS
(
    SELECT
        P.commit_id,
        P.entity_id,
        F.file_id,
        E.name,
        F.filename,
        E.kind,
        P.end_row - P.start_row AS loc
    FROM presence P
    JOIN filenames F ON F.entity_id = P.entity_id
    JOIN levels L ON L.entity_id = P.entity_id
    JOIN entities E ON E.id = P.entity_id
    WHERE L.level = 1
    GROUP BY P.commit_id, P.entity_id
),
member_counts AS
(
    SELECT
        P.commit_id,
        E.parent_id AS entity_id,
        COUNT(DISTINCT P.entity_id) AS member_count
    FROM presence P
    JOIN levels L ON L.entity_id = P.entity_id
    JOIN entities E ON E.id = P.entity_id
    WHERE L.level = 2
    GROUP BY P.commit_id, E.parent_id
),
fan_ins AS
(
    SELECT
        D.commit_id,
        TF.file_id,
        COUNT(DISTINCT SF.file_id) AS fan_in
    FROM deps D
    JOIN filenames SF ON SF.entity_id = D.src_id
    JOIN filenames TF ON TF.entity_id = D.tgt_id
    WHERE SF.file_id <> TF.file_id
    GROUP BY D.commit_id, TF.file_id
),
user_counts AS
(
    SELECT
        F.file_id,
        COUNT(DISTINCT CH.commit_id) AS commit_count,
        COUNT(DISTINCT author_email) AS author_count,
        COUNT(DISTINCT committer_email) AS committer_count,
        SUM(CH.adds) AS adds,
        SUM(CH.dels) AS dels,
        SUM(CH.adds) + SUM(CH.dels) AS churn
    FROM changes CH
    JOIN filenames F ON F.entity_id = CH.entity_id
    JOIN commits CO ON CO.id = CH.commit_id
    GROUP BY F.file_id
)
SELECT
    locs.file_id AS id,
    -- refs.name AS ref_name,
    -- DATE(commits.committer_date, 'unixepoch') AS ref_date,
    locs.filename,
    -- locs.kind,
    locs.loc,
    member_counts.member_count AS members,
    fan_ins.fan_in,
    user_counts.commit_count AS commits,
    user_counts.churn,
    user_counts.author_count AS authors
    -- user_counts.committer_count,
    -- user_counts.adds,
    -- user_counts.dels,
FROM locs
JOIN member_counts ON member_counts.commit_id = locs.commit_id AND member_counts.entity_id = locs.entity_id
JOIN fan_ins ON fan_ins.commit_id = locs.commit_id AND fan_ins.file_id = locs.file_id
JOIN user_counts ON user_counts.file_id = locs.file_id
JOIN refs ON refs.commit_id = locs.commit_id
JOIN commits ON commits.id = locs.commit_id
WHERE refs.name = :ref_name
-- AND locs.loc >= :min_locs
-- AND (user_counts.author_count > 1 OR user_counts.committer_count > 1)
AND locs.filename NOT LIKE '%/test/%'
AND locs.filename NOT LIKE '%/tests/%'
AND locs.filename NOT LIKE '%/testkit/%'
AND locs.filename NOT LIKE '%/generated/%'
AND locs.filename NOT LIKE 'test/%'
AND locs.filename NOT LIKE 'tests/%'
AND locs.filename NOT LIKE 'testkit/%'
AND locs.filename NOT LIKE 'generated/%'
ORDER BY locs.file_id