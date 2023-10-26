SELECT
    R.id,
    R.commit_id,
    DATE(C.committer_date, 'unixepoch') AS committer_date,
    R.name
FROM refs R
JOIN commits C ON C.id = R.commit_id
ORDER BY C.committer_date DESC