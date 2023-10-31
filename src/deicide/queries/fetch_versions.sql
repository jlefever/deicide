WITH p_commit_ids AS (
    SELECT P.commit_id
    FROM presence P
    GROUP BY P.commit_id
)
SELECT
    PCI.commit_id,
    C.committer_date AS date,
    C.sha1,
    R.name AS ref_name
FROM p_commit_ids PCI
JOIN commits C ON C.id = PCI.commit_id
LEFT JOIN refs R ON R.commit_id = PCI.commit_id
ORDER BY C.committer_date DESC