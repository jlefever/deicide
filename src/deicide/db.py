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


def fetch_candidate_files(con: Con, ref_name: str) -> pd.DataFrame:
    params = {"ref_name": ref_name}
    return pd.read_sql(_get_query_by_caller_name(), con, params=params)


def fetch_children(con: Con, ref_name: str, target_id: int) -> pd.DataFrame:
    params = {"ref_name": ref_name, "target_id": str(target_id)}
    return pd.read_sql(_get_query_by_caller_name(), con, index_col="id", params=params)


def fetch_client_deps(con: Con, target_id: int, target_file: str) -> pd.DataFrame:
    params = {"target_id": str(target_id), "target_file": target_file}
    return pd.read_sql(_get_query_by_caller_name(), con, params=params)


def fetch_clients(con: Con, target_file: str) -> pd.DataFrame:
    params = {"target_file": target_file}
    return pd.read_sql(_get_query_by_caller_name(), con, index_col="id", params=params)


def fetch_entities_by_name(con: Con, name: str) -> pd.DataFrame:
    params = {"name": name}
    return pd.read_sql(_get_query_by_caller_name(), con, params=params)


def fetch_internal_deps(con: Con, target_id: int) -> pd.DataFrame:
    params = {"target_id": str(target_id)}
    return pd.read_sql(_get_query_by_caller_name(), con, params=params)


def fetch_outgoing_type_names(con: Con, target_id: int) -> list[str]:
    params = {"target_id": str(target_id)}
    return list(pd.read_sql(_get_query_by_caller_name(), con, params=params)["name"])


def fetch_refs(con: Con) -> pd.DataFrame:
    return pd.read_sql(_get_query_by_caller_name(), con)


def fetch_touches(con: Con, ref_name: str, target_id: str) -> pd.DataFrame:
    params = {"ref_name": ref_name, "target_id": str(target_id)}
    return pd.read_sql(_get_query_by_caller_name(), con, params=params)


def fetch_lead_ref_name(con: Con) -> str:
    return fetch_refs(con).iloc[0]["name"]
