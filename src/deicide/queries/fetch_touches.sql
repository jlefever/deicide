SELECT CO.sha1, CO.author_email, CH.entity_id, CH.adds, CH.dels
FROM changes CH
JOIN presence P ON P.entity_id = CH.entity_id
JOIN refs R ON R.commit_id = P.commit_id
JOIN entities E ON E.id = CH.entity_id
JOIN commits CO ON CO.id = CH.commit_id
WHERE R.name = :ref_name AND E.parent_id = :target_id
ORDER BY CO.author_email, CO.committer_date, E.id