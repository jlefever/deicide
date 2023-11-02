import abc
import random
import math
import itertools as it
from functools import cache

# from typing import Generic, TypeVar

from deicide.validation2 import ClusterPath, Clustering, to_alpha


class Cluster(abc.ABC):
    @cache
    def members(self) -> list[int]:
        return self._members()

    @abc.abstractmethod
    def _members(self) -> list[int]:
        return

    @abc.abstractmethod
    def to_clustering(self) -> Clustering:
        pass

    # @abc.abstractmethod
    # def accept(self, visitor: "ClusterVisitor[T]") -> "T":
    #     return


class Singleton(Cluster):
    def __init__(self, item_id: int):
        self._item_id = item_id

    @property
    def item_id(self):
        return self._item_id

    def _members(self) -> list[int]:
        return [self.item_id]

    def to_clustering(self) -> Clustering:
        return Clustering([(self.item_id, ClusterPath(str(self.item_id)))])

    # def accept(self, visitor: "ClusterVisitor[T]") -> "T":
    #     return visitor.visit_singleton(self)


class Composite(Cluster):
    def __init__(self, clusters: list[Cluster]):
        if len(clusters) < 2:
            raise ValueError("composite cluster must contain at least two elements")
        self._clusters = clusters

    @property
    def clusters(self) -> list[Cluster]:
        return self._clusters

    def _members(self) -> list[int]:
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

    @cache
    def __call__(self, a: Cluster, b: Cluster) -> float:
        pairs = it.product(a.members(), b.members())
        dists = [self._dist(u, v) for (u, v) in pairs]
        return self._aggregate(dists)

    @abc.abstractmethod
    def _aggregate(self, dists: list[float]) -> float:
        return


class MinLinkDist(LinkDist):
    def _aggregate(self, dists: list[float]) -> float:
        return min(dists)


class AvgLinkDist(LinkDist):
    def _aggregate(self, dists: list[float]) -> float:
        return sum(dists) / len(dists)


class MaxLinkDist(LinkDist):
    def _aggregate(self, dists: list[float]) -> float:
        return max(dists)


def hac(
    ids: set[int],
    link_dist: LinkDist,
    threshold: float = math.inf,
    seed: int | None = None,
) -> Cluster:
    random.seed(seed)
    clusters: list[Cluster] = [Singleton(id) for id in ids]
    if len(clusters) == 0:
        raise ValueError("need at least one id")
    while len(clusters) > 1:
        pairs = it.combinations(clusters, 2)
        dists = [((a, b), link_dist(a, b)) for (a, b) in pairs]
        # Shuffle the list to get a random minimum (if there are multiple)
        random.shuffle(dists)
        (a, b), min_dist = min(dists, key=lambda x: x[1])
        if all(d == min_dist for _, d in dists):
            return Composite(clusters)
        if min_dist >= threshold:
            return Composite(clusters)
        clusters.remove(a)
        clusters.remove(b)
        clusters.append(Composite([a, b]))
    return clusters[0]
