SELECT E.id, E.parent_id, E.name, E.kind, E.disc
FROM entities E
JOIN presence P ON P.entity_id = E.id
WHERE E.name = :name AND P.commit_id = :commit_id