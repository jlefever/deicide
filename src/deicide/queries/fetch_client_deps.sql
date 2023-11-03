SELECT SF.file_id AS src_id, D.tgt_id, D.kind
FROM deps D
JOIN temp.filenames SF ON SF.entity_id = D.src_id
JOIN entities TE ON TE.id = D.tgt_id
WHERE 
    TE.parent_id = :target_id AND
    SF.file_id <> :file_id AND
    D.commit_id = :commit_id
GROUP BY SF.file_id, D.tgt_id, D.kind