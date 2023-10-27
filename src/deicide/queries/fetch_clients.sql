SELECT
    SF.file_id AS id,
    NULL AS parent_id,
    SF.filename AS name,
    'file' AS kind
FROM deps D
JOIN temp.filenames SF  ON SF.entity_id = D.src_id
JOIN temp.filenames TF  ON TF.entity_id = D.tgt_id
WHERE
    SF.file_id <> TF.file_id AND
    TF.file_id = :file_id AND
    D.commit_id = :commit_id
GROUP BY SF.file_id, SF.filename