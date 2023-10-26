SELECT E.id, E.parent_id, E.name, E.kind, E.disc
FROM entities E
WHERE E.name = :name