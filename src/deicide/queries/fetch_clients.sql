SELECT DISTINCT SFE.id, SFE.parent_id, SFE.name, SFE.kind
FROM deps D
JOIN temp.filenames SF  ON SF.entity_id = D.src_id
JOIN temp.filenames TF  ON TF.entity_id = D.tgt_id
JOIN entities       SFE ON SFE.id = SF.file_id
WHERE
    SF.filename <> TF.filename AND
    TF.filename = :target_file