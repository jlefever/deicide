SELECT D.src_id, D.tgt_id, D.kind
FROM deps D
JOIN entities SE ON SE.id = D.src_id
JOIN entities TE ON TE.id = D.tgt_id
WHERE
    SE.parent_id = :target_id AND
    TE.parent_id = :target_id AND
    D.src_id <> D.tgt_id