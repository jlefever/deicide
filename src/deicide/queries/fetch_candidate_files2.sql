SELECT
    P.commit_id,
    P.entity_id AS file_id,
    E.name AS filename,
    P.end_row - P.start_row AS loc
FROM presence P
JOIN entities E ON E.id = P.entity_id
WHERE E.kind == 'file' AND P.commit_id = :commit_id
GROUP BY P.commit_id, P.entity_id
ORDER BY (P.end_row - P.start_row) DESC
LIMIT :n