import inspect
import sqlite3
from pathlib import Path

import pandas as pd

Con = sqlite3.Connection

QUERIES_PATH = Path(__file__).absolute().parent.joinpath("queries")


def _get_query(query_name: str) -> str:
    return QUERIES_PATH.joinpath(f"{query_name}.sql").read_text()


def _get_query_by_caller_name() -> str:
    func_name = inspect.stack()[1][3]
    return _get_query(func_name)


def create_temp_tables(con: Con):
    con.executescript(_get_query("_prelude"))


def fetch_candidate_files(con: Con, commit_id: int) -> pd.DataFrame:
    params = {"commit_id": commit_id}
    return pd.read_sql(_get_query_by_caller_name(), con, params=params)


def fetch_candidate_files2(con: Con, commit_id: int, n: int) -> pd.DataFrame:
    params = {"commit_id": commit_id, "n": n}
    return pd.read_sql(_get_query_by_caller_name(), con, params=params)


def fetch_children(con: Con, commit_id: int, target_id: int) -> pd.DataFrame:
    params = {"commit_id": commit_id, "target_id": str(target_id)}
    return pd.read_sql(_get_query_by_caller_name(), con, index_col="id", params=params)


def fetch_client_deps(
    con: Con, commit_id: int, target_id: int, file_id: str
) -> pd.DataFrame:
    params = {
        "commit_id": commit_id,
        "target_id": str(target_id),
        "file_id": file_id,
    }
    return pd.read_sql(_get_query_by_caller_name(), con, params=params)


def fetch_clients(con: Con, commit_id: int, file_id: int) -> pd.DataFrame:
    params = {"commit_id": commit_id, "file_id": file_id}
    return pd.read_sql(_get_query_by_caller_name(), con, index_col="id", params=params)


def fetch_entities_by_name(con: Con, commit_id: int, name: str) -> pd.DataFrame:
    params = {"commit_id": commit_id, "name": name}
    return pd.read_sql(_get_query_by_caller_name(), con, params=params)


def fetch_internal_deps(con: Con, commit_id: int, target_id: int) -> pd.DataFrame:
    params = {"commit_id": commit_id, "target_id": str(target_id)}
    return pd.read_sql(_get_query_by_caller_name(), con, params=params)


def fetch_refs(con: Con) -> pd.DataFrame:
    return pd.read_sql(_get_query_by_caller_name(), con)


def fetch_touches(con: Con, target_id: str) -> pd.DataFrame:
    params = {"target_id": str(target_id)}
    return pd.read_sql(_get_query_by_caller_name(), con, params=params)


def fetch_versions(con: Con) -> pd.DataFrame:
    return pd.read_sql(_get_query_by_caller_name(), con, index_col="commit_id")
