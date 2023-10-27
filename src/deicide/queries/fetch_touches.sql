SELECT CO.sha1, CO.author_email, CH.entity_id, CH.adds, CH.dels
FROM changes CH
JOIN entities E ON E.id = CH.entity_id
JOIN commits CO ON CO.id = CH.commit_id
WHERE E.parent_id = :target_id
ORDER BY CO.author_email, CO.committer_date, E.id