SELECT E.id, E.parent_id, E.name, E.kind, P.start_row, P.end_row, P.identifiers
FROM presence P
JOIN entities E ON E.id = P.entity_id
WHERE P.commit_id = :commit_id AND parent_id = :target_id
ORDER BY P.start_row, E.name, E.id