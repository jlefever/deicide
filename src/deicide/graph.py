from typing import Iterable, Callable

import numpy as np
import pandas as pd
from ordered_set import OrderedSet as oset
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def no_loops(edges: oset[tuple[int, int]]) -> oset[tuple[int, int]]:
    return oset((a, b) for a, b in edges if a != b)


def map_edges(edges: oset[tuple[int, int]], f: Callable[[int], int]) -> oset[tuple[int, int]]:
    return oset((f(a), f(b)) for a, b in edges)


def group_edges_by(edges: oset[tuple[int, int]], col: pd.Series) -> oset[tuple[int, int]]:
    return map_edges(edges, lambda x: col.loc[x])


def to_adj(nodes: oset[int], edges: oset[tuple[int, int]]) -> np.ndarray:
    arr = np.zeros((len(nodes), len(nodes)))
    for src, tgt in edges:
        src_ix = nodes.index(src)
        tgt_ix = nodes.index(tgt)
        arr[src_ix, tgt_ix] = 1.0
    return arr


def to_sc_components(adj: np.ndarray) -> list[int]:
    _, labels = connected_components(
        csr_matrix(adj), connection="strong", directed=True, return_labels=True
    )
    return list(labels)


def to_wc_components(adj: np.ndarray) -> list[int]:
    _, labels = connected_components(
        csr_matrix(adj), connection="weak", directed=True, return_labels=True
    )
    return list(labels)


def group_by_scc(seq: Iterable[int], edges: oset[tuple[int, int]]) -> list[int]:
    nodes = oset(seq)
    comp_labels = to_sc_components(to_adj(nodes, edges))
    return [comp_labels[nodes.index(e)] for e in seq]


def group_by_wcc(seq: Iterable[int], edges: oset[tuple[int, int]]) -> list[int]:
    nodes = oset(seq)
    comp_labels = to_wc_components(to_adj(nodes, edges))
    return [comp_labels[nodes.index(e)] for e in seq]