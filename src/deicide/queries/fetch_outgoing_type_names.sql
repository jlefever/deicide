SELECT DISTINCT TE.name
FROM deps D
JOIN entities SE ON SE.id = D.src_id
JOIN entities TE ON TE.id = D.tgt_id
WHERE
	SE.parent_id = :target_id AND (
  	TE.kind = 'class' OR
  	TE.kind = 'interface' OR
  	TE.kind = 'enum' OR
  	TE.kind = 'annotation')
ORDER BY TE.name
