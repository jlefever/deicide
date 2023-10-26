CREATE TEMP TABLE temp.ancestors AS
WITH RECURSIVE ancestors (entity_id, ancestor_id) AS
(
    SELECT E.id AS entity_id, E.id AS ancestor_id
    FROM entities E

    UNION ALL

    SELECT E.id AS entity_id, A.ancestor_id
    FROM ancestors A
    JOIN entities E ON A.entity_id = E.parent_id
)
SELECT * FROM ancestors;

CREATE TEMP TABLE temp.filenames AS
SELECT E.id AS entity_id, FE.id AS file_id, FE.name AS filename
FROM entities E
JOIN temp.ancestors A ON A.entity_id = E.id
JOIN entities FE ON FE.id = A.ancestor_id
WHERE FE.parent_id IS NULL;

CREATE TEMP TABLE IF NOT EXISTS temp.levels AS
WITH RECURSIVE levels (entity_id, level) AS
(
    SELECT E.id AS entity_id, 0 as level
    FROM entities E
    WHERE E.parent_id IS NULL

    UNION ALL

    SELECT E.id AS entity_id, L.level + 1
    FROM entities E, levels L
    WHERE E.parent_id = L.entity_id
)
SELECT * FROM levels;