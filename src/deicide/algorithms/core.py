import abc
import math
import random
import itertools as it
from functools import cache
from typing import Iterable

from deicide.validation2 import ClusterPath, Clustering, to_alpha


def jaccard(a: set[int], b: set[int]) -> float:
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


class Cluster(abc.ABC):
    @cache
    def members(self) -> frozenset[int]:
        return frozenset(self._members())

    @abc.abstractmethod
    def _members(self) -> Iterable[int]:
        return

    @abc.abstractmethod
    def to_clustering(self) -> Clustering:
        return


class Leaf(Cluster):
    def __init__(self, item_ids: list[int]) -> None:
        if len(item_ids) < 1:
            raise ValueError("leaf cluster must contain at least one item")
        self._item_ids = item_ids

    @property
    def item_ids(self) -> list[int]:
        return self._item_ids

    def _members(self) -> Iterable[int]:
        return self.item_ids

    def to_clustering(self) -> Clustering:
        return Clustering(
            (e, ClusterPath(to_alpha(i))) for i, e in enumerate(self.item_ids)
        )

    # def accept(self, visitor: "ClusterVisitor[T]") -> "T":
    #     return visitor.visit_singleton(self)


class Branch(Cluster):
    def __init__(self, clusters: list[Cluster]):
        if len(clusters) < 2:
            raise ValueError(
                "branch cluster must contain at least two children clusters"
            )
        self._clusters = clusters

    @property
    def clusters(self) -> list[Cluster]:
        return self._clusters

    def _members(self) -> Iterable[int]:
        members: list[int] = []
        for cluster in self.clusters:
            members.extend(cluster.members())
        return members

    def to_clustering(self) -> Clustering:
        res = Clustering([])
        for i, cluster in enumerate(self.clusters):
            root = ClusterPath(to_alpha(i))
            clustering = cluster.to_clustering().with_root(root)
            res = res.union(clustering)
        return res

    # def accept(self, visitor: "ClusterVisitor[T]") -> "T":
    #     return visitor.visit_composite(self)


# T = TypeVar("T")


# class ClusterVisitor(abc.ABC, Generic[T]):
#     @abc.abstractmethod
#     def visit_singleton(self, singleton: Singleton) -> T:
#         return

#     @abc.abstractmethod
#     def visit_composite(self, composite: Composite) -> T:
#         return


# class ClusterNamer(ClusterVisitor[None]):
#     def __init__(self):
#         self._stack = []

#     def visit_singleton(self, singleton: Singleton) -> None:
#         return super().visit_singleton(singleton)

#     def visit_composite(self, composite: Composite) -> None:
#         return super().visit_composite(composite)


class Dist(abc.ABC):
    @abc.abstractmethod
    def __call__(self, a: int, b: int) -> float:
        return


class LinkDist(abc.ABC):
    def __init__(self, dist: Dist):
        self._dist = dist

    def find_min(
        self,
        clusters_X: Iterable[Cluster],
        clusters_Y: Iterable[Cluster] | None = None,
        *,
        shuffle: bool,
    ) -> tuple[tuple[Cluster, Cluster], float, bool]:
        if clusters_Y is None or clusters_X == clusters_Y:
            pairs = it.combinations(clusters_X, 2)
        else:
            pairs = it.product(clusters_X, clusters_Y)
        dists = [((a, b), self(a, b)) for (a, b) in pairs]
        # Shuffle the list to get a random minimum (if there are multiple)
        if shuffle:
            random.shuffle(dists)
        (a, b), min_dist = min(dists, key=lambda x: x[1])
        all_equal = all(d == min_dist for _, d in dists)
        return ((a, b), min_dist, all_equal)

    @cache
    def __call__(self, a: Cluster, b: Cluster) -> float:
        if a == b:  # by ref
            pairs = it.combinations(a.members(), 2)
        else:
            pairs = it.product(a.members(), b.members())
        return self._aggregate(self._dist(u, v) for (u, v) in pairs)

    @abc.abstractmethod
    def _aggregate(self, dists: Iterable[float]) -> float:
        return


class MinLinkDist(LinkDist):
    def _aggregate(self, dists: Iterable[float]) -> float:
        return min(dists)


class AvgLinkDist(LinkDist):
    def _aggregate(self, dists: Iterable[float]) -> float:
        total, n = 0.0, 0
        for dist in dists:
            total += dist
            n += 1
        return total / n


class MaxLinkDist(LinkDist):
    def _aggregate(self, dists: Iterable[float]) -> float:
        return max(dists)


def hac(
    ids: set[int],
    link_dist: LinkDist,
    *,
    shuffle: bool,
    threshold: float = math.inf,
) -> Cluster:
    if len(ids) == 0:
        raise ValueError("need at least one id")
    clusters: list[Cluster] = [Leaf([id]) for id in ids]
    while len(clusters) > 1:
        (a, b), min_dist, _ = link_dist.find_min(clusters, shuffle=shuffle)
        if min_dist >= threshold:
            return Branch(clusters)
        clusters.remove(a)
        clusters.remove(b)
        clusters.append(Branch([a, b]))
    return clusters[0]


def greedy_hac(
    ids: set[int],
    link_dist: LinkDist,
    *,
    shuffle: bool,
    threshold: float = math.inf,
) -> Cluster:
    if len(ids) == 0:
        raise ValueError("need at least one id")
    clusters: list[Cluster] = []
    singletons: list[Cluster] = [Leaf([id]) for id in ids]
    while len(singletons) > 1:
        (a, b), min_dist, _ = link_dist.find_min(singletons, shuffle=shuffle)
        if min_dist > threshold:
            break
        root = Branch([a, b])
        singletons.remove(a)
        singletons.remove(b)
        while len(singletons) > 0:
            (a, b), min_dist, _ = link_dist.find_min(
                [root], singletons, shuffle=shuffle
            )
            if min_dist > threshold:
                break
            root = Branch([root, b])
            singletons.remove(b)
        clusters.append(root)
    res = clusters + singletons
    if len(res) > 1:
        return Branch(clusters + singletons)
    else:
        return res[0]
