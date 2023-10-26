SELECT E.id, E.parent_id, E.name, E.kind, P.start_row, P.end_row
FROM presence P
JOIN refs R ON R.commit_id = P.commit_id
JOIN entities E ON E.id = P.entity_id
WHERE R.name = :ref_name AND parent_id = :target_id
ORDER BY P.start_row, E.name, E.id