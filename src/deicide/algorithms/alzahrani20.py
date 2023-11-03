import itertools as it
import math
from statistics import median
from functools import cache, cached_property

from deicide.algorithms import core
from deicide.validation2 import Clustering
from deicide.loading import Dataset


class GodClass:
    def __init__(self, methods_to_clients: dict[int, set[int]]):
        self._methods_to_clients = methods_to_clients

    @cached_property
    def methods(self) -> set[int]:
        return set(self._methods_to_clients.keys())

    @property
    def clients(self) -> dict[int, set[int]]:
        return self._methods_to_clients

    def clients_of(self, method: int) -> set[int]:
        return self.clients[method]

    @cache
    def sim(self, method_u: int, method_v: int) -> float:
        return core.jaccard(self.clients_of(method_u), self.clients_of(method_v))

    def dist(self, method_u: int, method_v: int) -> float:
        return 1.0 - self.sim(method_u, method_v)

    def get_recommended_threshold(self) -> float:
        # Assumes at least one method
        dists = (self.dist(u, v) for u, v in it.combinations(self.methods, 2))
        dists = [d for d in dists if d < 1.0]
        if len(dists) == 0:
            return math.inf
        return median(dists)

    # def sim_with_set(self, method: int, others: set[int]) -> float:
    #     return sum(self.sim(method, o) for o in others) / len(others)


class AlzahraniDist(core.Dist):
    def __init__(self, god_class: GodClass):
        self._god_class = god_class

    def __call__(self, a: int, b: int) -> float:
        return self._god_class.dist(a, b)


def merge_small_clusters(
    branch: core.Branch, link_dist: core.LinkDist, *, min_size: int = 2
) -> core.Branch:
    # I appologize for this God awful implementation.
    # Note: The behavior seems to depend on the order of the clusters. I am not
    # sure if this desired but I want to stay true to the oringal paper.
    # https://www.mdpi.com/2076-3417/10/17/6038
    to_be_removed: list[core.Cluster] = []
    to_be_added: list[core.Cluster] = []
    for cluster in branch.clusters:
        if len(cluster.members()) >= min_size:
            continue
        if cluster in to_be_removed:
            continue
        remaining_clusters = list(branch.clusters)
        remaining_clusters.remove(cluster)
        remaining_clusters.extend(to_be_added)
        for c in to_be_removed:
            remaining_clusters.remove(c)
        min_dist, min_cluster = math.inf, None
        dists: list[float] = []
        for remaining_cluster in remaining_clusters:
            candidate = core.Branch([remaining_cluster, cluster])
            dist = link_dist(candidate, candidate)
            dists.append(dist)
            if dist < min_dist:
                min_dist = dist
                min_cluster = remaining_cluster
        all_equal = all(d == min_dist for d in dists)
        if min_cluster is not None and not all_equal:
            to_be_removed.append(min_cluster)
            to_be_removed.append(cluster)
            to_be_added.append(core.Branch([min_cluster, cluster]))
    res = list(branch.clusters)
    res.extend(to_be_added)
    for c in to_be_removed:
        res.remove(c)
    return core.Branch(res)


def alzahrani20(god_class: GodClass, *, shuffle: bool) -> Clustering:
    link_dist = core.AvgLinkDist(AlzahraniDist(god_class))
    cluster = core.greedy_hac(
        ids=god_class.methods,
        link_dist=link_dist,
        threshold=god_class.get_recommended_threshold(),
        shuffle=shuffle,
    )
    return merge_small_clusters(cluster, link_dist).to_clustering()


def to_godclass(ds: Dataset) -> GodClass:
    methods: list[int] = []
    for id, row in ds.targets_df.iterrows():  # type: ignore
        if row["kind"] == "method" or row["kind"] == "constructor":
            methods.append(int(id))  # type: ignore
    clients: dict[int, set[int]] = {m: set() for m in methods}
    for id, row in ds.client_deps_df.iterrows():  # type: ignore
        src_id, tgt_id = int(row["src_id"]), int(row["tgt_id"])  # type: ignore
        if tgt_id in clients:
            clients[tgt_id].add(src_id)
    return GodClass(clients)
