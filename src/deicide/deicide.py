import itertools as it
from collections import defaultdict
from math import ceil

from ortools.sat.python import cp_model

from deicide.core import Dep, Entity
from deicide.semantic import SemanticSimilarity

TEXT_EDGE_MULTIPLIER = 1


def deicide(
    targets: list[Entity],
    clients: list[Entity],
    deps: list[Dep],
    semantic: SemanticSimilarity,
) -> list[tuple[str, list[int]]]:
    # Set up graph
    entities = targets + clients
    id_to_ix: dict[str, int] = {e.id: ix for ix, e in enumerate(entities)}
    di_edges: set[tuple[int, int]] = set()
    for dep in deps:
        if dep.src_id != dep.tgt_id:
            di_edges.add((id_to_ix[dep.src_id], id_to_ix[dep.tgt_id]))

    # Create edges from name similarity
    un_weights: dict[tuple[int, int], float] = dict()
    for a_entity, b_entity in it.combinations(targets, 2):
        score = semantic.sim(a_entity.id, b_entity.id)
        if score <= 0:
            continue
        a_ix, b_ix = id_to_ix[a_entity.id], id_to_ix[b_entity.id]
        un_weights[a_ix, b_ix] = score
    un_edges = set(un_weights.keys())

    # Create node weights
    # Targets are weighted to 1 while clients are weighted to 0
    node_weights: dict[int, int] = dict()
    for ix in range(0, len(targets)):
        node_weights[ix] = 1
    for ix in range(len(targets), len(entities)):
        node_weights[ix] = 0

    # Create edge weights
    edge_weights: dict[tuple[int, int], float] = defaultdict(float)
    for src, tgt in di_edges:
        edge_weights[src, tgt] = 1.0
    for a, b in un_edges:
        edge_weights[a, b] += un_weights[a, b] * TEXT_EDGE_MULTIPLIER

    # Run clustering algorithm
    nodes = set(range(len(entities)))
    res = _recursive_partition(nodes, di_edges, un_edges, node_weights, edge_weights)

    # Return clustering
    memberships: list[tuple[str, list[int]]] = []
    for ix, entity in enumerate(entities):
        memberships.append((entity.id, res[ix]))
    return memberships


def _recursive_partition(
    nodes: set[int],
    di_edges: set[tuple[int, int]],
    un_edges: set[tuple[int, int]],
    node_weights: dict[int, int],
    edge_weights: dict[tuple[int, int], float],
    k: int = 2,
    eps: float = 0.1,
    min_node_weight: int = 2,
    max_time_in_seconds: float | None = 30,
) -> dict[int, list[int]]:
    """Recursively partition a graph that may have many components. The input
    graph can have both directed and undirected edges. This will keep dividing
    until min_node_weight is reached.

    Args:
        nodes (set[int]):  A sequence of nodes
        di_edges (set[tuple[int, int]]): Directed edges (ordered pairs) that may have
            cycles
        un_edges (set[tuple[int, int]]): Undirected edges (unordered pairs)
        node_weights (dict[int, int]): A mapping from nodes to weights
        edge_weights (dict[tuple[int, int], float]): A mapping from edges to weights
        k (int, optional): The number of desired clusters on each branch. Defaults to 2.
        eps (float, optional): Balance parameter. Defaults to 0.1.
        min_node_weight (int, optional): When to stop dividing. Defaults to 2.
        max_time_in_seconds (float | None, optional): When to stop searching for a
            better solution. Defaults to 30.

    Returns:
        dict[int, list[int]]: A mapping from nodes to their cluster path
    """
    # Map nodes to their strongly connected components (SCCs) This results in an
    # acyclic directed graph (DAG) called the "condensation graph". Only
    # directed edges can be used for this.
    node_to_scc = _find_scc(nodes, di_edges)

    # Map SCCs to the nodes that they contain
    scc_to_nodes: dict[int, set[int]] = defaultdict(set)
    for node, scc in node_to_scc.items():
        scc_to_nodes[scc].add(node)

    # Create weights for each SCC by summing their nodes
    scc_weights: dict[int, int] = dict()
    for scc, nodes in scc_to_nodes.items():
        scc_weights[scc] = sum(node_weights[n] for n in nodes)

    # Create edges between SCCs by aggregating the edges between nodes
    scc_di_edges = {(node_to_scc[a], node_to_scc[b]) for a, b in di_edges}
    scc_un_edges = {(node_to_scc[a], node_to_scc[b]) for a, b in un_edges}

    # Create weights for the edges between SCCs by taking the max
    # Note: The max corresponds to "complete linkage". Alternatively, we could
    # use min or mean which would correspond to "single" and "average" linkage
    # respectively. We do not want to use the sum as it is not
    # "scale-invariant"; in other words, it would give much higher priority to
    # large WCCs.
    scc_edge_weights: dict[tuple[int, int], float] = defaultdict(float)
    for (a, b), w in edge_weights.items():
        scc_a, scc_b = node_to_scc[a], node_to_scc[b]
        scc_edge_weights[scc_a, scc_b] = max(scc_edge_weights[scc_a, scc_b], w)

    # We can now partition the condensation graph as it is a DAG
    scc_to_path = _recursive_partition_dag(
        set(node_to_scc.values()),
        scc_di_edges,
        scc_un_edges,
        scc_weights,
        scc_edge_weights,
        k,
        eps,
        min_node_weight,
        max_time_in_seconds,
    )

    # Map the clusters of SCCs to the original nodes and return
    node_to_path: dict[int, list[int]] = dict()
    for node, scc in node_to_scc.items():
        node_to_path[node] = scc_to_path[scc]
    return node_to_path


def _recursive_partition_dag(
    nodes: set[int],
    di_edges: set[tuple[int, int]],
    un_edges: set[tuple[int, int]],
    node_weights: dict[int, int],
    edge_weights: dict[tuple[int, int], float],
    k: int,
    eps: float,
    min_node_weight: int,
    max_time_in_seconds: float | None,
) -> dict[int, list[int]]:
    """Recursively partition a directed acyclic graph (DAG) that may have many
    components. The input graph can have both directed and undirected edges. This
    will keep dividing until min_node_weight is reached.

    Args:
        nodes (set[int]): A sequence of nodes
        di_edges (set[tuple[int, int]]): Directed edges (ordered pairs) with no cycles
        un_edges (set[tuple[int, int]]): Undirected edges (unordered pairs)
        node_weights (dict[int, int]): A mapping from nodes to weights
        edge_weights (dict[tuple[int, int], float]): A mapping from edges to weights
        k (int): The number of desired clusters on each branch
        eps (float): Balance parameter
        min_node_weight: When to stop dividing
        max_time_in_seconds (float | None): When to stop searching for a better solution

    Returns:
        dict[int, list[int]]: A mapping from nodes to their cluster path
    """
    # Map nodes to their weakly connected components (WCCs). There are no edges
    # between different WCCs.
    node_to_wcc = _find_wcc(nodes, di_edges)

    # TODO: We are currently using only directed edges to find WCCs. Why not use
    # all edges? Uncomment the following to use all edges.
    # node_to_wcc = _find_wcc(nodes, di_edges | un_edges)

    # Map WCCs to the nodes that they contain
    wcc_to_nodes: dict[int, set[int]] = defaultdict(set)
    for node, wcc in node_to_wcc.items():
        wcc_to_nodes[wcc].add(node)

    # Create initial clustering paths using WCCs as root clusters
    node_to_path: dict[int, list[int]] = dict()
    for node in nodes:
        node_to_path[node] = [node_to_wcc[node]]

    # Partition each WCC separately
    for wcc, component in wcc_to_nodes.items():
        paths = _recursive_partition_dag_component(
            {(a, b) for a, b in di_edges if a in component and b in component},
            {(a, b) for a, b in un_edges if a in component and b in component},
            node_weights,
            edge_weights,
            k,
            eps,
            min_node_weight,
            max_time_in_seconds,
        )
        for node, path in paths.items():
            node_to_path[node].extend(path)

    # Return the mapping from nodes to cluster paths
    return node_to_path


def _recursive_partition_dag_component(
    di_edges: set[tuple[int, int]],
    un_edges: set[tuple[int, int]],
    node_weights: dict[int, int],
    edge_weights: dict[tuple[int, int], float],
    k: int,
    eps: float,
    min_node_weight: int,
    max_time_in_seconds: float | None,
) -> dict[int, list[int]]:
    """Recursively partition a single component of a directed acyclic graph (DAG). The
    input graph can have both directed and undirected edges. This will keep dividing
    until min_node_weight is reached.

    The input graph must both be:
    - Acyclic: No cycles among the directed edges
    - Connected: All nodes must be reachable from each using either directed or
    undirected edges (i.e., a single component)

    Args:
        di_edges (set[tuple[int, int]]): Directed edges (ordered pairs) with no cycles
        un_edges (set[tuple[int, int]]): Undirected edges (unordered pairs)
        node_weights (dict[int, int]): A mapping from nodes to weights
        edge_weights (dict[tuple[int, int], float]): A mapping from edges to weights
        k (int): The number of desired clusters on each branch
        eps (float): Balance parameter
        min_node_weight: When to stop dividing
        max_time_in_seconds (float | None): When to stop searching for a better solution

    Returns:
        dict[int, list[int]]: A mapping from nodes to their cluster path
    """
    # Create an initial (empty) mapping from nodes to cluster paths
    paths: dict[int, list[int]] = defaultdict(list)

    # Define a recursive helper function that appends to the cluster paths
    def visit(nodes: set[int]) -> None:
        if sum(node_weights[n] for n in nodes) < min_node_weight:
            return
        node_to_labels = _partition_dag_component(
            di_edges,
            un_edges,
            {n: w if n in nodes else 0 for n, w in node_weights.items()},
            edge_weights,
            k,
            eps,
            max_time_in_seconds,
        )
        if node_to_labels is None:
            return
        label_to_nodes: dict[int, set[int]] = defaultdict(set)
        for node, label in node_to_labels.items():
            if node not in nodes:
                continue
            paths[node].append(label)
            label_to_nodes[label].add(node)
        for block in label_to_nodes.values():
            visit(block)

    # Call the helper with all nodes and return
    visit(set(it.chain(*(di_edges | un_edges))))
    return paths


def _partition_dag_component(
    di_edges: set[tuple[int, int]],
    un_edges: set[tuple[int, int]],
    node_weights: dict[int, int],
    edge_weights: dict[tuple[int, int], float],
    k: int,
    eps: float,
    max_time_in_seconds: float | None,
) -> dict[int, int] | None:
    """Partition a single component of a directed acyclic graph (DAG) into
    exactly k clusters. The input graph can have both directed and undirected
    edges. This function is not recursive (it will not keep dividing).

    The input graph must both be:
    - Acyclic: No cycles among the directed edges
    - Connected: All nodes must be reachable from each using either directed or
    undirected edges (i.e., a single component)

    Based on the work of Ozkaya and Catalyurek: https://arxiv.org/abs/2207.13638

    Args:
        di_edges (set[tuple[int, int]]): Directed edges (ordered pairs) with no cycles
        un_edges (set[tuple[int, int]]): Undirected edges (unordered pairs)
        node_weights (dict[int, int]): A mapping from nodes to weights
        edge_weights (dict[tuple[int, int], float]): A mapping from edges to weights
        k (int): The number of desired clusters
        eps (float): Balance parameter
        max_time_in_seconds (float | None): When to stop searching for a better solution

    Returns:
        dict[int, int] | None: A mapping from nodes to their cluster label or None if no
        solution was found
    """
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
    bound = ceil((1 + eps) * ceil(sum(node_weights[i] for i in nodes) / k))

    # Setup the constraint programming (CP) model
    model = cp_model.CpModel()

    # Variable: x_is indicates that node i is assigned to part s
    x: dict[tuple[int, int], cp_model.IntVar] = dict()
    for i in nodes:
        for s in parts:
            x[i, s] = model.NewBoolVar(f"x[{i},{s}]")

    # Variable: y_st indicates that there is an edge from part s to part t
    y: dict[tuple[int, int], cp_model.IntVar] = dict()
    for s in parts:
        for t in parts:
            if s != t:
                y[s, t] = model.NewBoolVar(f"y[{s},{t}]")

    # Variable: z_ij indicates that edge (i,j) is a cut edge
    z: dict[tuple[int, int], cp_model.IntVar] = dict()
    for i, j in edges:
        z[i, j] = model.NewBoolVar(f"z[{i},{j}]")

    # Create helper function to get edge weight as an integer
    def edge_weight(i: int, j: int) -> int:
        return round(edge_weights[i, j] * (1 << 16))

    # Objective: Minimize the edge cut.
    model.Minimize(sum(edge_weight(i, j) * z[i, j] for i, j in edges))

    # Constraint: All nodes must belong to exactly one part.
    for i in nodes:
        model.AddExactlyOne([x[i, s] for s in parts])

    # Constraint: All parts are be bounded in size.
    for s in parts:
        model.AddLinearConstraint(
            sum(node_weights[i] * x[i, s] for i in nodes), 1, bound
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
    node_to_label: dict[int, int] = dict()
    for i in nodes:
        for s in parts:
            if solver.BooleanValue(x[i, s]):
                node_to_label[i] = s
    return node_to_label


def _find_wcc(nodes: set[int], edges: set[tuple[int, int]]) -> dict[int, int]:
    """Finds the weakly connected components (WCCs) of an undirected or directed
    graph.

    Args:
        nodes (set[int]): A sequence of nodes
        edges (set[tuple[int, int]]): A sequence of undirected or directed edges

    Returns:
        dict[int, int]: A mapping from each node to its WCC
    """
    adj = _to_adj(edges | _transpose(edges))
    roots: dict[int, int] = dict()

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
    """Finds the strongly connected components (SCCs) of a directed graph using
    Kosaraju's algorithm.

    Args:
        nodes (set[int]): A sequence of nodes
        edges (set[tuple[int, int]]): A sequence of directed edges

    Returns:
        dict[int, int]: A mapping from each node to it SCC
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

    roots: dict[int, int] = dict()

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
    """Transposes the input sequences of edges

    Args:
        edges (set[tuple[int, int]]): A sequence of directed edges

    Returns:
        set[tuple[int, int]]: The transposed edges
    """
    return {(tgt, src) for src, tgt in edges}


def _to_adj(edges: set[tuple[int, int]]) -> defaultdict[int, set[int]]:
    """Produces an adjacency matrix from the input sequence of edges.

    Args:
        edges (set[tuple[int, int]]): A sequence of directed or undirected edges

    Returns:
        defaultdict[int, set[int]]: A mapping from nodes to their adjacent nodes
    """
    adj: defaultdict[int, set[int]] = defaultdict(set)
    for src, tgt in edges:
        if src == tgt:
            continue
        adj[src].add(tgt)
    return adj
