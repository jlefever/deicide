from collections import Counter
from dataclasses import dataclass
from functools import cache, cached_property
from genericpath import commonprefix
from itertools import combinations
from typing import Any, Iterable


@dataclass
class Entity:
    id: str
    parent_id: str | None
    name: str
    kind: str


@dataclass
class Dep:
    src_id: str
    tgt_id: str
    kind: str


_CLUSTER_PATH_SEP = "$"


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
    def parent(self) -> "ClusterPath | None":
        if len(self) <= 1:
            return None
        return self[0 : len(self) - 1]

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
    _pairs: frozenset[tuple[str, ClusterPath]]

    def __init__(self, pairs: Iterable[tuple[str, ClusterPath]]) -> None:
        self._pairs = frozenset(pairs)

    @property
    def pairs(self) -> frozenset[tuple[str, ClusterPath]]:
        return self._pairs

    def union(self, other: "Clustering") -> "Clustering":
        return Clustering(self.pairs | other.pairs)

    def subset(self, entities: set[str]) -> "Clustering":
        return Clustering((e, c) for e, c in self.pairs if e in entities)

    def truncate(self, level: int) -> "Clustering":
        return Clustering((e, c.truncate(level)) for e, c in self.pairs)

    def expand(self) -> "Clustering":
        expanded: set[tuple[str, ClusterPath]] = set()
        for entity, cluster in self.pairs:
            for ancestor in cluster.ancestors_with_self():
                expanded.add((entity, ancestor))
        return Clustering(expanded)

    def contract(self) -> "Clustering":
        pairs: list[tuple[str, ClusterPath]] = []
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

    def with_isolated(self, entities: Iterable[str]) -> "Clustering":
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
    def entities(self) -> set[str]:
        return {e for e, _ in self.pairs}

    @cache
    def clusters(self) -> set[ClusterPath]:
        return {c for _, c in self.pairs}

    @cache
    def singleton_clusters(self) -> dict[ClusterPath, str]:
        counter = Counter(c for (_, c) in self.pairs)
        singletons = (c for c in self.clusters() if counter[c] == 1)
        return {c: _single(self.entities_for(c)) for c in singletons}

    @cache
    def singleton_entities(self) -> dict[str, ClusterPath]:
        counter = Counter(e for (e, _) in self.pairs)
        singletons = (e for e in self.entities() if counter[e] == 1)
        return {c: _single(self.clusters_for(c)) for c in singletons}

    @cache
    def num_levels(self) -> int:
        return max(len(c) for c in self.clusters())

    def entities_for(self, cluster: ClusterPath) -> set[str]:
        return self.c2e[cluster]

    def clusters_for(self, entity: str) -> set[ClusterPath]:
        return self.e2c[entity]

    @cached_property
    def c2e(self) -> dict[ClusterPath, set[str]]:
        c2e: dict[ClusterPath, set[str]] = {c: set() for c in self.clusters()}
        for e, c in self.pairs:
            c2e[c].add(e)
        return c2e

    @cached_property
    def e2c(self) -> dict[str, set[ClusterPath]]:
        e2c: dict[str, set[ClusterPath]] = {e: set() for e in self.entities()}
        for e, c in self.pairs:
            e2c[e].add(c)
        return e2c


def _single(iterable: Iterable[Any]) -> Any:
    lst = list(iterable)
    if len(lst) != 1:
        raise ValueError(f"non-singleton ({len(lst)} elements)")
    return lst[0]
