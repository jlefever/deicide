import sqlite3
import os
from dataclasses import dataclass

import pandas as pd
from scipy.stats.mstats import rankdata

from deicide import db, jdeo


@dataclass
class Dataset:
    targets_df: pd.DataFrame
    target_deps_df: pd.DataFrame
    clients_df: pd.DataFrame
    client_deps_df: pd.DataFrame
    touches_df: pd.DataFrame

    def entities_df(self) -> pd.DataFrame:
        return pd.concat([self.targets_df, self.clients_df])

    def deps_df(self) -> pd.DataFrame:
        return pd.concat([self.target_deps_df, self.client_deps_df])    


def load_dataset(db_path: str, filename: str, commit_id: int) -> Dataset:
    with sqlite3.connect(db_path) as con:
        db.create_temp_tables(con)
        files_df = db.fetch_entities_by_name(con, commit_id, filename)
        files_df = files_df[files_df["kind"] == "file"]
        if len(files_df) < 1:
            raise RuntimeError(f"No files named '{filename}' found")
        if len(files_df) > 1:
            raise RuntimeError(f"Too many files named '{filename}' found")
        file_id = int(files_df.iloc[0]["id"])
        targets_df = db.fetch_children(con, commit_id, file_id)
        top_id = file_id
        # If there is only one top level item (e.g. a Java class), skip to its children
        if len(targets_df) == 1:
            top_id = int(targets_df.index[0])
            targets_df = db.fetch_children(con, commit_id, top_id)
        target_deps_df = db.fetch_internal_deps(con, commit_id, top_id)
        clients_df = db.fetch_clients(con, commit_id, top_id, file_id)
        client_deps_df = db.fetch_client_deps(con, commit_id, top_id, file_id)
        touches_df = db.fetch_touches(con, top_id)
        return Dataset(
            targets_df,
            target_deps_df,
            clients_df,
            client_deps_df,
            touches_df,
        )


def try_identify_version(db_path: str, version: str) -> int | None:
    with sqlite3.connect(db_path) as con:
        df = db.fetch_versions(con)
    by_ref = df[df["ref_name"] == version]
    if len(by_ref) > 0:
        assert len(by_ref) == 1 # DB constraint prevents this
        return int(by_ref.index[0])
    by_sha1 = df[df["sha1"] == version]
    if len(by_sha1) > 0:
        assert len(by_sha1) == 1 # DB constraint prevents this
        return int(by_sha1.index[0])
    return None


def fetch_latest_commit_id(db_path: str) -> int:
    with sqlite3.connect(db_path) as con:
        df = db.fetch_versions(con)
        return int(df.sort_values("date", ascending=False).index[0])


def load_jdeo_candidates(jdeo_dir: str) -> set[tuple[str, str]]:
    jdeo_candidates = set()

    for csv_name in list(sorted(os.listdir(jdeo_dir))):
        if not csv_name.endswith(".csv"):
            continue
        project_name = csv_name.split(".")[0]
        rows = jdeo.load_jdeo_project(jdeo_dir, project_name)
        jdeo_candidates |= {(project_name, r.filename) for r in rows}

    return jdeo_candidates


def load_candidates_df(db_dir: str) -> pd.DataFrame:
    candidates_dfs = []

    for db_name in list(sorted(os.listdir(db_dir))):
        if not db_name.endswith(".db"):
            continue
        project_name = db_name.split(".")[0]
        print(f"Finding candidates in {db_name}...")
        with sqlite3.connect(os.path.join(db_dir, db_name)) as con:
            db.create_temp_tables(con)
            versions_df = db.fetch_versions(con)
            commit_id = int(versions_df.sort_values("date", ascending=False).index[0])
            project_candidates_df = db.fetch_candidate_files(con, commit_id)
            project_candidates_df["id"] = project_candidates_df["id"].map(lambda x: f"{project_name}-{x}")
            project_candidates_df.insert(1, "commit_id", commit_id)
            project_candidates_df.insert(1, "project", project_name)
            candidates_dfs.append(project_candidates_df)

    candidates_df = pd.concat(candidates_dfs, ignore_index=True).set_index("id")
    return candidates_df.sort_values(["loc", "members", "fan_in"], ascending=False)


def append_jdeo_col(candidates_df: pd.DataFrame, jdeo_candidates: set[tuple[str, str]]):
    in_jdeo_data_col = []
    for _, row in candidates_df.iterrows():
        in_jdeo_data_col.append((row["project"], row["filename"]) in jdeo_candidates)
    candidates_df["in_jdeo_data"] = in_jdeo_data_col


def percentiles(arr):
    return (rankdata(arr) / len(arr)) * 100


def append_percentile_cols(candidates_df: pd.DataFrame):
    candidates_df["loc_pct"] = None
    candidates_df["members_pct"] = None
    candidates_df["fan_in_pct"] = None
    candidates_df["commits_pct"] = None
    candidates_df["churn_pct"] = None
    candidates_df["authors_pct"] = None

    for _, ix in candidates_df.groupby("project").groups.items():
        candidates_df.loc[ix, "loc_pct"] = percentiles(candidates_df.loc[ix]["loc"])
        candidates_df.loc[ix, "members_pct"] = percentiles(candidates_df.loc[ix]["members"])
        candidates_df.loc[ix, "fan_in_pct"] = percentiles(candidates_df.loc[ix]["fan_in"])
        candidates_df.loc[ix, "commits_pct"] = percentiles(candidates_df.loc[ix]["commits"])
        candidates_df.loc[ix, "churn_pct"] = percentiles(candidates_df.loc[ix]["churn"])
        candidates_df.loc[ix, "authors_pct"] = percentiles(candidates_df.loc[ix]["authors"])


def load_full_candidates_df(db_dir: str, jdeo_dir: str) -> pd.DataFrame:
    candidates_df = load_candidates_df(db_dir)
    jdeo_candidates = load_jdeo_candidates(jdeo_dir)
    append_percentile_cols(candidates_df)
    append_jdeo_col(candidates_df, jdeo_candidates)
    return candidates_df