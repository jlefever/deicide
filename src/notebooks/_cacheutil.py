"""Utility functions for caching. Only intended to be used from Jupyter notebooks."""

import pickle
from pathlib import Path
from typing import Callable

import pandas as pd

from deicide.loading import Dataset
from deicide.validation2 import Clustering

CACHE_DIR = Path("cache")


def to_path(cache_key: str) -> Path:
    return Path(CACHE_DIR, f"{cache_key}.csv")


def save(cache_key: str, df: pd.DataFrame):
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(to_path(cache_key))


def load(cache_key: str, index_col: int | str | None = 0) -> pd.DataFrame:
    return pd.read_csv(to_path(cache_key), index_col=index_col)  # type: ignore


def is_cached(cache_key: str) -> bool:
    return to_path(cache_key).exists()


def load_or_create(cache_key: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    if is_cached(cache_key):
        return load(cache_key)
    df = fn()
    save(cache_key, df)
    return df


def save_dataset_to_cache(cache_key: str, ds: Dataset):
    save(f"dataset-{cache_key}-targets", ds.targets_df)
    save(f"dataset-{cache_key}-target-deps", ds.target_deps_df)
    save(f"dataset-{cache_key}-clients", ds.clients_df)
    save(f"dataset-{cache_key}-client-deps", ds.client_deps_df)
    save(f"dataset-{cache_key}-touches", ds.touches_df)


def load_dataset_from_cache(cache_key: str) -> Dataset:
    targets_df = load(f"dataset-{cache_key}-targets", "id")
    target_deps_df = load(f"dataset-{cache_key}-target-deps")
    clients_df = load(f"dataset-{cache_key}-clients", "id")
    client_deps_df = load(f"dataset-{cache_key}-client-deps")
    touches_df = load(f"dataset-{cache_key}-touches")
    return Dataset(
        targets_df,
        target_deps_df,
        clients_df,
        client_deps_df,
        touches_df,
    )


def is_dataset_cached(cache_key: str) -> bool:
    a = is_cached(f"dataset-{cache_key}-targets")
    b = is_cached(f"dataset-{cache_key}-target-deps")
    c = is_cached(f"dataset-{cache_key}-clients")
    d = is_cached(f"dataset-{cache_key}-client-deps")
    e = is_cached(f"dataset-{cache_key}-touches")
    return all([a, b, c, d, e])


def to_clustering_path(cache_key: str) -> Path:
    return Path(CACHE_DIR, f"clustering-{cache_key}.bin")


def is_clustering_cached(cache_key: str) -> bool:
    return to_clustering_path(cache_key).exists()


def save_clustering(cache_key: str, clustering: Clustering):
    with open(to_clustering_path(cache_key), "wb") as file:
        pickle.dump(clustering, file)


def load_clustering(cache_key: str) -> Clustering:
    with open(to_clustering_path(cache_key), "rb") as file:
        return pickle.load(file)


def load_or_create_clustering(
    cache_key: str, fn: Callable[[], Clustering]
) -> Clustering:
    if is_clustering_cached(cache_key):
        return load_clustering(cache_key)
    clustering = fn()
    save_clustering(cache_key, clustering)
    return clustering
