from math import ceil, inf
from typing import Callable

from ortools.sat.python import cp_model


def partition(
    edges: list[tuple[int, int]],
    w: Callable[[int], int],
    c: Callable[[int, int], int],
    k: int,
    eps: float,
) -> tuple[float, dict[int, int] | None]:
    # Remove any self-edges or duplicates
    edges = list(sorted({(a, b) for a, b in edges if a != b}))

    # Create a list of node ids found in the edge list
    nodes = list(sorted({a for a, _ in edges} | {b for _, b in edges}))

    # Create a list of partition ids
    parts = list(range(k))

    # Calculate the upper bound on partition size
    bound = ceil((1 + eps) * ceil(sum(w(i) for i in nodes) / k))

    # Setup the constraint programming (CP) model
    model = cp_model.CpModel()

    # Variable: x_is indicates that node i is assigned to part s
    x = {}
    for i in nodes:
        for s in parts:
            x[i, s] = model.NewBoolVar(f"x[{i},{s}]")

    # Variable: y_st indicates that there is an edge from part s to part t
    y = {}
    for s in parts:
        for t in parts:
            if s != t:
                y[s, t] = model.NewBoolVar(f"y[{s},{t}]")

    # Variable: z_ij indicates that edge (i,j) is a cut edge
    z = {}
    for i, j in edges:
        z[i, j] = model.NewBoolVar(f"z[{i},{j}]")

    # Objective: Minimize the edge cut.
    model.Minimize(sum(c(i, j) * z[i, j] for i, j in edges))

    # Constraint: All nodes must belong to exactly one part.
    for i in nodes:
        model.AddExactlyOne([x[i, s] for s in parts])

    # Constraint: No part must be larger than a certain bound.
    for s in parts:
        model.AddLinearConstraint(sum(w(i) * x[i, s] for i in nodes), 0, bound)

    # Constraint: Mark the cut edges as one if they are in different parts.
    for i, j in edges:
        for s in parts:
            model.Add(x[j, s] - x[i, s] <= z[i, j])

    # Constraint: Mark the adjacency of parts for cut edges.
    for i, j in edges:
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
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model)

    # Check if successful
    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        return 0.0, None

    # Extract labels into dictionary
    labels = {}
    for i in nodes:
        for s in parts:
            if solver.BooleanValue(x[i, s]):
                labels[i] = s
    return solver.ObjectiveValue(), labels


def partition2(
    di_edges: set[tuple[int, int]],
    un_edges: set[tuple[int, int]],
    node_weight: Callable[[int], int],
    edge_weight: Callable[[int, int], int],
    k: int,
    eps: float,
    max_time_in_seconds: float | None = None,
) -> tuple[float, dict[int, int] | None]:
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
    x = {}
    for i in nodes:
        for s in parts:
            x[i, s] = model.NewBoolVar(f"x[{i},{s}]")

    # Variable: y_st indicates that there is an edge from part s to part t
    y = {}
    for s in parts:
        for t in parts:
            if s != t:
                y[s, t] = model.NewBoolVar(f"y[{s},{t}]")

    # Variable: z_ij indicates that edge (i,j) is a cut edge
    z = {}
    for i, j in edges:
        z[i, j] = model.NewBoolVar(f"z[{i},{j}]")

    # Objective: Minimize the edge cut.
    model.Minimize(sum(edge_weight(i, j) * z[i, j] for i, j in edges))

    # Constraint: All nodes must belong to exactly one part.
    for i in nodes:
        model.AddExactlyOne([x[i, s] for s in parts])

    # Constraint: No part must be larger than a certain bound.
    for s in parts:
        model.AddLinearConstraint(
            sum(node_weight(i) * x[i, s] for i in nodes), 0, bound
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
    if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
        return inf, None

    # Extract labels into dictionary
    labels = {}
    for i in nodes:
        for s in parts:
            if solver.BooleanValue(x[i, s]):
                labels[i] = s
    return solver.ObjectiveValue(), labels
