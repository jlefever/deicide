from os.path import commonprefix
from functools import cache, cached_property
from collections import Counter
from typing import Iterable, Any
from itertools import combinations

import numpy as np
import pandas as pd
from ordered_set import OrderedSet as oset

from deicide.jdeo import JDeoRow


_CLUSTER_PATH_SEP = "$"


def single(iterable: Iterable[Any]) -> Any:
    lst = list(iterable)
    if len(lst) != 1:
        raise ValueError(f"non-singleton ({len(lst)} elements)")
    return lst[0]


class ClusterPath:
    _segments: list[str]

    def __init__(self, segments: Iterable[str]):
        if isinstance(segments, str):
            self._segments = [segments]
        else:
            self._segments = list(segments)
        if any(_CLUSTER_PATH_SEP in s for s in self._segments):
            raise ValueError(f"cluster name has illegal char: '{_CLUSTER_PATH_SEP}'")

    @property
    def segments(self) -> list[str]:
        return self._segments

    @cached_property
    def name(self) -> str:
        return _CLUSTER_PATH_SEP.join(self.segments)

    @cache
    def ancestors(self) -> list["ClusterPath"]:
        return [self[0:i] for i in range(1, len(self))]

    @cache
    def ancestors_with_self(self) -> list["ClusterPath"]:
        return self.ancestors() + [self]

    def truncate(self, level: int) -> "ClusterPath":
        return self[0 : level + 1]

    def fill(self, placeholder: str, length: int) -> "ClusterPath":
        return ClusterPath(self.segments + ([placeholder] * (length - len(self))))

    def is_ancestor_of(self, other: "ClusterPath") -> bool:
        if len(self) >= len(other):
            return False
        return all(a == b for a, b in zip(self.segments, other.segments))

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __le__(self, other: "ClusterPath") -> bool:
        return self.segments <= other.segments

    def __lt__(self, other: "ClusterPath") -> bool:
        return self.segments < other.segments

    def __ge__(self, other: "ClusterPath") -> bool:
        return self.segments >= other.segments

    def __gt__(self, other: "ClusterPath") -> bool:
        return self.segments > other.segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, ix: slice) -> "ClusterPath":
        return ClusterPath(self.segments[ix])

    def __repr__(self) -> str:
        return f"ClusterPath({self.name})"


class Clustering:
    _pairs: frozenset[tuple[int, ClusterPath]]

    def __init__(self, pairs: Iterable[tuple[int, ClusterPath]]):
        self._pairs = frozenset(pairs)

    @property
    def pairs(self) -> frozenset[tuple[int, ClusterPath]]:
        return self._pairs

    def union(self, other: "Clustering") -> "Clustering":
        return Clustering(self.pairs | other.pairs)

    def subset(self, entities: set[int]) -> "Clustering":
        return Clustering((e, c) for e, c in self.pairs if e in entities)

    def truncate(self, level: int) -> "Clustering":
        return Clustering((e, c.truncate(level)) for e, c in self.pairs)

    def expand(self) -> "Clustering":
        expanded = set()
        for entity, cluster in self.pairs:
            for ancestor in cluster.ancestors_with_self():
                expanded.add((entity, ancestor))
        return Clustering(expanded)

    def contract(self) -> "Clustering":
        pairs = []
        for entity in self.entities():
            clusters = sorted(self.clusters_for(entity))  # shorter elements come first
            excluded = {a for a, b in combinations(clusters, 2) if a.is_ancestor_of(b)}
            pairs.extend((entity, c) for c in clusters if c not in excluded)
        return Clustering(pairs)

    def without_singletons(self) -> "Clustering":
        clustering = self.expand()
        counter = Counter(c for (_, c) in clustering.pairs)
        clustering = Clustering((e, c) for e, c in clustering.pairs if counter[c] > 1)
        return clustering.contract()

    def fill(self, placeholder: str) -> "Clustering":
        return Clustering(
            (e, c.fill(placeholder, self.num_levels())) for e, c in self.pairs
        )

    def per_level(self) -> list["Clustering"]:
        return [self.truncate(i) for i in range(self.num_levels())]

    def root(self) -> ClusterPath:
        return ClusterPath(commonprefix([c.segments for c in self.clusters()]))

    def without_root(self) -> "Clustering":
        prefix_len = len(self.root())
        return Clustering((e, c[prefix_len:]) for e, c in self.pairs)

    def with_root(self, root: ClusterPath) -> "Clustering":
        return Clustering(
            (e, ClusterPath(root.segments + c.segments)) for e, c in self.pairs
        )

    def replace_root(self, root: ClusterPath) -> "Clustering":
        return self.without_root().with_root(root)

    def with_isolated(self, entities: Iterable[int]) -> "Clustering":
        # This does not consider the rare case where an existing cluster
        # happens to have the same name as an entity id 
        isolated = sorted(set(entities) - self.entities())
        pairs = {(e, ClusterPath(str(e))) for e in isolated}
        return Clustering(pairs | self.pairs)

    def normalize(self) -> "Clustering":
        clustering = self.without_root()
        entities = clustering.entities()
        # Remove any cluster that both
        # 1) contains only a single entity
        # 2) is not that entity's only cluster
        return clustering.without_singletons().with_isolated(entities)

    @cache
    def entities(self) -> set[int]:
        return {e for e, _ in self.pairs}

    @cache
    def clusters(self) -> set[ClusterPath]:
        return {c for _, c in self.pairs}

    @cache
    def singleton_clusters(self) -> dict[ClusterPath, int]:
        counter = Counter(c for (_, c) in self.pairs)
        singletons = (c for c in self.clusters() if counter[c] == 1)
        return {c: single(self.entities_for(c)) for c in singletons}

    @cache
    def singleton_entities(self) -> dict[int, ClusterPath]:
        counter = Counter(e for (e, _) in self.pairs)
        singletons = (e for e in self.entities() if counter[e] == 1)
        return {c: single(self.clusters_for(c)) for c in singletons}

    @cache
    def num_levels(self) -> int:
        return max(len(c) for c in self.clusters())

    def entities_for(self, cluster: ClusterPath) -> set[int]:
        return self.c2e[cluster]

    def clusters_for(self, entity: int) -> set[ClusterPath]:
        return self.e2c[entity]

    @cached_property
    def c2e(self) -> dict[ClusterPath, set[int]]:
        c2e = {c: set() for c in self.clusters()}
        for e, c in self.pairs:
            c2e[c].add(e)
        return c2e

    @cached_property
    def e2c(self) -> dict[int, set[ClusterPath]]:
        e2c = {e: set() for e in self.entities()}
        for e, c in self.pairs:
            e2c[e].add(c)
        return e2c

    @cache
    def ndarray(self) -> np.ndarray:
        clusters = oset(sorted(self.clusters()))
        entities = oset(sorted(self.entities()))
        arr = np.zeros((len(clusters), len(entities)))
        for entity, cluster in self.pairs:
            row_ix = clusters.index(cluster)
            col_ix = entities.index(entity)
            arr[row_ix, col_ix] = 1.0
        return arr


def to_alpha(num: int) -> str:
    if num < 0:
        raise ValueError("num must be nonnegative")
    chars: list[str] = []
    num = num + 1
    while num > 0:
        mod = (num - 1) % 26
        chars.append(chr(65 + mod))
        num = (num - mod) // 26
    return "".join(reversed(chars))


def to_my_clustering(entities_df: pd.DataFrame) -> Clustering:
    triples = ((id, r["block_name"], r["kind"]) for id, r in entities_df.iterrows())
    return Clustering(
        (e, ClusterPath(c.split(".")))
        for e, c, k in triples
        if k != "file"  # type: ignore
    )


def to_my_clustering_with_files(entities_df: pd.DataFrame) -> Clustering:
    pairs = ((id, r["block_name"]) for id, r in entities_df.iterrows())
    return Clustering((e, ClusterPath(c.split("."))) for e, c in pairs)  # type: ignore


def to_jdeo_clustering(jdeo_id_map: dict[JDeoRow, int]) -> Clustering:
    return Clustering(
        (id, ClusterPath(row.cluster.split(".")))
        for row, id in jdeo_id_map.items()
        if id != None
    )


def to_commit_clustering(touches_df: pd.DataFrame) -> Clustering:
    return Clustering(
        (r["entity_id"], ClusterPath([r["sha1"]])) for _, r in touches_df.iterrows()
    )


def to_author_clustering(touches_df: pd.DataFrame) -> Clustering:
    return Clustering(
        (r["entity_id"], ClusterPath([r["author_email"]]))
        for _, r in touches_df.iterrows()
    )


def to_client_clustering(
    clients_df: pd.DataFrame, client_deps_df: pd.DataFrame
) -> Clustering:
    memberships = set()
    for _, row in client_deps_df.iterrows():
        src_id = int(row["src_id"])
        tgt_id = int(row["tgt_id"])
        client = str(clients_df.loc[src_id]["name"])
        memberships.add((tgt_id, ClusterPath([client])))
    return Clustering(memberships)
