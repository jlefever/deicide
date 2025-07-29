import sqlite3
from pathlib import Path

from deicide.core import Dep, Entity


class DbDriver:
    def __init__(self, path: Path) -> None:
        """
        Initializes the database driver with a connection to the given SQLite database file.
        """
        self._conn = sqlite3.connect(path)
        self._cursor = self._conn.cursor()
        self._commit_hash = None

    def set_commit_hash(self, commit_hash: str) -> None:
        """
        Sets the active commit hash to filter dependency queries.
        Use "WORKDIR" to indicate uncommitted changes (NULL commit_id).
        """
        if commit_hash == "WORKDIR":
            self._commit_hash = None
        else:
            self._commit_hash = commit_hash

    def load_commit_hashes(self) -> list[str]:
        """
        Returns all unique commit hashes in the database.
        Entries with NULL commit_id are labeled as "WORKDIR".
        """
        self._cursor.execute("SELECT DISTINCT commit_id FROM deps")
        hashes = (row[0] for row in self._cursor.fetchall())
        return [h if h else "WORKDIR" for h in hashes]

    def load_filenames(self) -> dict[str, str]:
        """
        Returns a mapping of file names to their corresponding entity IDs (hex-encoded),
        limited to entities of kind 'File'.
        """
        # TODO: I'm not sure what happens when there are many versions. Will it
        # subtly break?
        sql = "SELECT name, id FROM entities WHERE parent_id IS NULL"
        return {r[0]: r[1].hex() for r in self._cursor.execute(sql)}

    def load_children(self, entity_id: str) -> list[Entity]:
        """
        Returns the direct children of the entity identified by the given ID.
        """
        sql = """
            SELECT id, parent_id, name, kind
            FROM entities WHERE parent_id = ?
            ORDER BY start_byte
        """
        self._cursor.execute(sql, (bytes.fromhex(entity_id),))
        rows = self._cursor.fetchall()
        return [self._make_entity(*row) for row in rows]

    def load_internal_deps(self, parent_id: str) -> list[Dep]:
        """
        Returns dependencies between sibling entities that share the same parent.
        Excludes self-dependencies and filters by the current commit hash.
        """
        sql = """
            SELECT D.src, D.tgt, D.kind
            FROM deps D
            JOIN entities SE ON SE.id = D.src
            JOIN entities TE ON TE.id = D.tgt
            WHERE
                SE.parent_id = :pid AND
                TE.parent_id = :pid AND
                D.src <> D.tgt AND
                (D.commit_id IS :cid OR D.commit_id = :cid)
        """
        pid = bytes.fromhex(parent_id)
        cid = None if self._commit_hash is None else bytes.fromhex(self._commit_hash)
        self._cursor.execute(sql, {"pid": pid, "cid": cid})
        return [self._make_dep(*r) for r in self._cursor.fetchall()]

    def load_clients(self, parent_id: str) -> list[Entity]:
        """
        Returns the root entities that depend on any direct child of the given parent entity.

        A root entity is defined as one with no parent (i.e., a top-level entity in the hierarchy).
        This method excludes roots that belong to the same hierarchy as the given parent_id.
        Only dependencies targeting direct children of the parent are considered.
        """
        sql = """
            WITH parent_root AS (
                SELECT root_id FROM temp.roots WHERE entity_id = :pid
            )
            SELECT DISTINCT
                RE.id,
                RE.parent_id,
                RE.name,
                RE.kind
            FROM deps D
            JOIN temp.roots SR ON SR.entity_id = D.src
            JOIN entities RE ON RE.id = SR.root_id
            WHERE D.tgt IN (
                SELECT id FROM entities WHERE parent_id = :pid
            )
            AND (D.commit_id IS :cid OR D.commit_id = :cid)
            AND SR.root_id != (SELECT root_id FROM parent_root)
            ORDER BY name
        """
        pid = bytes.fromhex(parent_id)
        cid = None if self._commit_hash is None else bytes.fromhex(self._commit_hash)
        self._ensure_roots_table()
        self._cursor.execute(sql, {"pid": pid, "cid": cid})
        rows = self._cursor.fetchall()
        return [self._make_entity(*r) for r in rows]

    def load_client_deps(self, parent_id: str) -> list[Dep]:
        """
        Returns dependencies where the source is the root entity of a different
        hierarchy that depends (possibly transitively via its descendants) on a
        direct child of the given parent entity.

        The root of the parent entity's hierarchy is excluded to avoid
        self-dependencies. Filters dependencies by the current commit hash.
        """
        sql = """
            WITH parent_root AS (
                SELECT root_id FROM temp.roots WHERE entity_id = :pid
            )
            SELECT DISTINCT
                R.root_id AS src,
                E.id AS tgt,
                D.kind
            FROM entities E
            JOIN deps D ON D.tgt = E.id
            JOIN temp.roots R ON D.src = R.entity_id
            WHERE E.parent_id = :pid
            AND (D.commit_id IS :cid OR D.commit_id = :cid)
            AND R.root_id != (SELECT root_id FROM parent_root)
        """
        pid = bytes.fromhex(parent_id)
        cid = None if self._commit_hash is None else bytes.fromhex(self._commit_hash)
        self._ensure_roots_table()
        self._cursor.execute(sql, {"pid": pid, "cid": cid})
        return [self._make_dep(*r) for r in self._cursor.fetchall()]

    def _ensure_roots_table(self) -> None:
        """
        Ensures that the temporary 'roots' table exists, which maps each entity
        to its root ancestor (an entity with no parent).
        """
        self._cursor.execute("""
            SELECT name FROM sqlite_temp_master
            WHERE type='table' AND name='roots'
        """)
        if self._cursor.fetchone():
            return
        self._cursor.execute("""
            CREATE TEMP TABLE temp.roots AS
            WITH RECURSIVE ancestors(entity_id, ancestor_id) AS (
                SELECT E.id, E.id
                FROM entities E
                UNION ALL
                SELECT E.id, A.ancestor_id
                FROM ancestors A
                JOIN entities E ON A.entity_id = E.parent_id
            )
            SELECT E.id AS entity_id, FE.id AS root_id
            FROM entities E
            JOIN ancestors A ON A.entity_id = E.id
            JOIN entities FE ON FE.id = A.ancestor_id
            WHERE FE.parent_id IS NULL
        """)

    def _make_entity(
        self, id: bytes, parent_id: bytes | None, name: str, kind: str
    ) -> Entity:
        """
        Constructs an Entity object from database row fields.
        """
        return Entity(
            id=id.hex(),
            parent_id=parent_id.hex() if parent_id is not None else None,
            name=name,
            kind=kind,
        )

    def _make_dep(self, src_id: bytes, tgt_id: bytes, kind: str) -> Dep:
        """
        Constructs a Dep object from database row fields.
        """
        return Dep(
            src_id=src_id.hex(),
            tgt_id=tgt_id.hex(),
            kind=kind,
        )

    def close(self) -> None:
        """
        Closes the database connection.
        """
        self._conn.close()
