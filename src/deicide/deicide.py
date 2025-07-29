from collections import defaultdict
from enum import Enum
from itertools import product
from math import ceil
from typing import Callable

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import IntVar

from deicide.core import Dep, Entity


def deicide(
    targets: list[Entity],
    clients: list[Entity],
    internal_deps: list[Dep],
    client_deps: list[Dep],
) -> list[tuple[str, list[int]]]:
    # Set up graph
    entities = targets + clients
    id_to_ix: dict[str, int] = {e.id: ix for ix, e in enumerate(entities)}
    di_edges: set[tuple[int, int]] = set()
    for dep in internal_deps:
        di_edges.add((id_to_ix[dep.src_id], id_to_ix[dep.tgt_id]))
    for dep in client_deps:
        di_edges.add((id_to_ix[dep.src_id], id_to_ix[dep.tgt_id]))

    def get_node_weight(ix: int) -> int:
        return 1 if ix < len(targets) else 0

    def get_edge_weight(src_ix: int, tgt_ix: int) -> int:
        return 1

    # Run cluster algorithm
    nodes = set(range(len(entities)))
    res = _cluster(nodes, di_edges, set(), get_node_weight, get_edge_weight)

    # Return clustering
    # memberships: list[tuple[str, ClusterPath]] = []
    # for ix, entity in enumerate(targets):
    #     path = ClusterPath([f"M{label}" for label in res[ix]])
    #     memberships.append((entity.id, path))
    # return Clustering(memberships)

    # Return clustering
    memberships: list[tuple[str, list[int]]] = []
    for ix, entity in enumerate(targets):
        memberships.append((entity.id, res[ix]))
    return memberships


class _Linkage(Enum):
    SINGLE = 0
    AVERAGE = 1
    COMPLETE = 2


def _cluster(
    nodes: set[int],
    di_edges: set[tuple[int, int]],
    un_edges: set[tuple[int, int]],
    node_weight: Callable[[int], int],
    edge_weight: Callable[[int, int], int],
    k: int = 2,
    eps: float = 0.1,
    linkage: _Linkage = _Linkage.COMPLETE,
    min_node_weight: int = 2,
    max_time_in_seconds: float | None = 30,
) -> dict[int, list[int]]:
    node_to_scc = _find_scc(nodes, di_edges)
    scc_to_nodes: dict[int, set[int]] = defaultdict(set)
    for node, scc in node_to_scc.items():
        scc_to_nodes[scc].add(node)

    strong_nodes = set(scc_to_nodes.keys())

    strong_di_edges = {(node_to_scc[a], node_to_scc[b]) for a, b in di_edges}
    strong_di_edges = {(a, b) for a, b in strong_di_edges if a != b}

    strong_un_edges = {(node_to_scc[a], node_to_scc[b]) for a, b in un_edges}
    strong_un_edges = {(a, b) for a, b in strong_un_edges if a != b}

    def strong_node_weight(scc: int) -> int:
        return sum(node_weight(node) for node in scc_to_nodes[scc])

    if linkage == _Linkage.SINGLE:

        def strong_edge_weight(scc_a: int, scc_b: int) -> int:
            a_nodes = scc_to_nodes[scc_a]
            b_nodes = scc_to_nodes[scc_b]
            return min(edge_weight(a, b) for a, b in product(a_nodes, b_nodes))

    elif linkage == _Linkage.AVERAGE:

        def strong_edge_weight(scc_a: int, scc_b: int) -> int:
            a_nodes = scc_to_nodes[scc_a]
            b_nodes = scc_to_nodes[scc_b]
            values = [edge_weight(a, b) for a, b in product(a_nodes, b_nodes)]
            return round(sum(values) / len(values))

    elif linkage == _Linkage.COMPLETE:

        def strong_edge_weight(scc_a: int, scc_b: int) -> int:
            a_nodes = scc_to_nodes[scc_a]
            b_nodes = scc_to_nodes[scc_b]
            return max(edge_weight(a, b) for a, b in product(a_nodes, b_nodes))

    else:
        raise RuntimeError

    scc_to_path = _cluster_dag(
        strong_nodes,
        strong_di_edges,
        strong_un_edges,
        strong_node_weight,
        strong_edge_weight,
        k,
        eps,
        min_node_weight,
        max_time_in_seconds,
    )

    node_to_path: dict[int, list[int]] = dict()
    for node, scc in node_to_scc.items():
        node_to_path[node] = scc_to_path[scc]
    return node_to_path


def _cluster_dag(
    nodes: set[int],
    di_edges: set[tuple[int, int]],
    un_edges: set[tuple[int, int]],
    node_weight: Callable[[int], int],
    edge_weight: Callable[[int, int], int],
    k: int,
    eps: float,
    min_node_weight: int,
    max_time_in_seconds: float | None,
) -> dict[int, list[int]]:
    # Cluster nodes by their weakly connected component
    edges = {(a, b) for a, b in di_edges | un_edges if edge_weight(a, b) > 0}
    node_to_wcc = _find_wcc(nodes, edges)

    # Create initial clustering paths using the weakly connected component as
    # the root cluster
    node_to_path: dict[int, list[int]] = dict()
    for node in nodes:
        node_to_path[node] = [node_to_wcc[node]]

    # Use the RecursivePartitioner on each weakly connected component
    wcc_to_nodes: dict[int, set[int]] = defaultdict(set)
    for node, wcc in node_to_wcc.items():
        wcc_to_nodes[wcc].add(node)
    for wcc, component in wcc_to_nodes.items():
        partitioner = _RecursivePartitioner(
            {(a, b) for a, b in di_edges if a in component and b in component},
            {(a, b) for a, b in un_edges if a in component and b in component},
            node_weight,
            edge_weight,
            k,
            eps,
            min_node_weight,
            max_time_in_seconds,
        )
        partitioner.partition(component)
        for node, path in partitioner.paths().items():
            node_to_path[node].extend(path)

    return node_to_path


def _find_wcc(nodes: set[int], edges: set[tuple[int, int]]) -> dict[int, int]:
    """
    Computes the weakly connected components.
    Returns a dict mapping each node to the root of its WCC.
    """
    adj = _to_adj(edges | _transpose(edges))
    roots: dict[int, int] = {}

    def assign(node: int, root: int) -> None:
        if node in roots:
            return
        roots[node] = root
        for neighbor in sorted(adj[node]):
            assign(neighbor, root)

    for i, node in enumerate(sorted(nodes, reverse=True)):
        assign(node, i)

    # Remap root IDs to be contiguous
    unique_roots = sorted(set(roots.values()))
    root_map = {old: new for new, old in enumerate(unique_roots)}
    for node in roots:
        roots[node] = root_map[roots[node]]

    return roots


def _find_scc(nodes: set[int], edges: set[tuple[int, int]]) -> dict[int, int]:
    """
    Computes the strongly connected components using Kosaraju's algorithm.
    Returns a dict mapping each node to the root of its SCC.
    """
    adj = _to_adj(edges)
    adj_inv = _to_adj(_transpose(edges))

    visited: set[int] = set()
    order: list[int] = []

    def visit(node: int) -> None:
        if node in visited:
            return
        visited.add(node)
        for neighbor in sorted(adj[node]):
            visit(neighbor)
        order.append(node)

    for node in sorted(nodes):
        visit(node)
    order.reverse()

    roots: dict[int, int] = {}

    def assign(node: int, root: int) -> None:
        if node in roots:
            return
        roots[node] = root
        for neighbor in sorted(adj_inv[node]):
            assign(neighbor, root)

    for i, node in enumerate(order):
        assign(node, i)

    # Remap root IDs to be contiguous
    unique_roots = sorted(set(roots.values()))
    root_map = {old: new for new, old in enumerate(unique_roots)}
    for node in roots:
        roots[node] = root_map[roots[node]]

    return roots


def _transpose(edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(tgt, src) for src, tgt in edges}


def _to_adj(edges: set[tuple[int, int]]) -> defaultdict[int, set[int]]:
    adj: defaultdict[int, set[int]] = defaultdict(set)
    for src, tgt in edges:
        if src == tgt:
            continue
        adj[src].add(tgt)
    return adj


class _RecursivePartitioner:
    def __init__(
        self,
        di_edges: set[tuple[int, int]],
        un_edges: set[tuple[int, int]],
        node_weight: Callable[[int], int],
        edge_weight: Callable[[int, int], int],
        k: int,
        eps: float,
        min_node_weight: int,
        max_time_in_seconds: float | None,
    ) -> None:
        self._paths: dict[int, list[int]] = defaultdict(list)
        self._di_edges = di_edges
        self._un_edges = un_edges
        self._node_weight = node_weight
        self._edge_weight = edge_weight
        self._k = k
        self._eps = eps
        self._min_node_weight = min_node_weight
        self._max_time_in_seconds = max_time_in_seconds

    def partition(self, nodes: set[int]) -> None:
        if sum(self._node_weight(x) for x in nodes) < self._min_node_weight:
            return
        node_to_labels = _partition(
            self._di_edges,
            self._un_edges,
            lambda x: self._node_weight(x) if x in nodes else 0,
            self._edge_weight,
            self._k,
            self._eps,
            self._max_time_in_seconds,
        )
        if node_to_labels is None:
            return
        label_to_nodes: dict[int, set[int]] = defaultdict(set)
        for node, label in node_to_labels.items():
            if node not in nodes:
                continue
            self._paths[node].append(label)
            label_to_nodes[label].add(node)
        for block in label_to_nodes.values():
            self.partition(block)

    def paths(self) -> dict[int, list[int]]:
        return self._paths


def _partition(
    di_edges: set[tuple[int, int]],
    un_edges: set[tuple[int, int]],
    node_weight: Callable[[int], int],
    edge_weight: Callable[[int, int], int],
    k: int,
    eps: float,
    max_time_in_seconds: float | None,
) -> dict[int, int] | None:
    # Remove any self-edges
    di_edges = {(a, b) for a, b in di_edges if a != b}
    un_edges = {(a, b) for a, b in un_edges if a != b}

    # An edge cannot be both directed and undirected
    un_edges = un_edges - di_edges

    # Create a set for all edges
    edges = di_edges | un_edges

    # Create a list of node ids found in the edge list
    nodes = list(sorted({a for a, _ in edges} | {b for _, b in edges}))

    # Create a list of partition ids
    parts = list(range(k))

    # Calculate the upper bound on partition size
    bound = ceil((1 + eps) * ceil(sum(node_weight(i) for i in nodes) / k))

    # Setup the constraint programming (CP) model
    model = cp_model.CpModel()

    # Variable: x_is indicates that node i is assigned to part s
    x: dict[tuple[int, int], IntVar] = {}
    for i in nodes:
        for s in parts:
            x[i, s] = model.NewBoolVar(f"x[{i},{s}]")

    # Variable: y_st indicates that there is an edge from part s to part t
    y: dict[tuple[int, int], IntVar] = {}
    for s in parts:
        for t in parts:
            if s != t:
                y[s, t] = model.NewBoolVar(f"y[{s},{t}]")

    # Variable: z_ij indicates that edge (i,j) is a cut edge
    z: dict[tuple[int, int], IntVar] = {}
    for i, j in edges:
        z[i, j] = model.NewBoolVar(f"z[{i},{j}]")

    # Objective: Minimize the edge cut.
    model.Minimize(sum(edge_weight(i, j) * z[i, j] for i, j in edges))

    # Constraint: All nodes must belong to exactly one part.
    for i in nodes:
        model.AddExactlyOne([x[i, s] for s in parts])

    # Constraint: All parts are be bounded in size.
    for s in parts:
        model.AddLinearConstraint(
            sum(node_weight(i) * x[i, s] for i in nodes), 1, bound
        )

    # Constraint: Mark the cut edges as one if they are in different parts.
    for i, j in edges:
        for s in parts:
            model.Add(x[j, s] - x[i, s] <= z[i, j])

    # Constraint: Mark the adjacency of parts for cut edges (only for directed edges).
    for i, j in di_edges:
        for s in parts:
            for t in parts:
                if s != t:
                    model.Add(x[i, s] + x[j, t] - 1 <= y[s, t])

    # Constraint: Force y to be triangular.
    for s in parts[1:]:
        for t in parts[:s]:
            model.Add(y[s, t] == 0)

    # Solve
    solver = cp_model.CpSolver()
    if max_time_in_seconds:
        solver.parameters.max_time_in_seconds = max_time_in_seconds
    status = solver.Solve(model)

    # Check if successful
    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:  # type: ignore
        return None

    # Extract labels into dictionary
    labels: dict[int, int] = {}
    for i in nodes:
        for s in parts:
            if solver.BooleanValue(x[i, s]):
                labels[i] = s
    return labels
