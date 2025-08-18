from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from deicide.core import Dep, Entity


@dataclass
class DV8ClusteringNode:
    """Represents a node in the DV8 clustering tree."""

    name: str
    type: Literal["group", "item"]
    nested: list[DV8ClusteringNode] | None = None

    def __post_init__(self) -> None:
        """Ensure that "item" nodes do not have nested children"""
        if self.type == "item" and self.nested is not None:
            raise ValueError("Item nodes cannot have nested children")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result: dict[str, Any] = {"@type": self.type, "name": self.name}
        if self.type == "group" and self.nested is not None:
            result["nested"] = [node.to_dict() for node in self.nested]
        return result


@dataclass
class DV8Clustering:
    """Root data structure for DV8 clustering format"""

    schema_version: str = "1.0"  # set default to 1.0
    name: str = ""
    structure: list[DV8ClusteringNode] = field(default_factory=list) # type: ignore

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "@schemaVersion": self.schema_version,
            "name": self.name,
            "structure": [node.to_dict() for node in self.structure],
        }


def create_dv8_clustering(
    clustering: list[tuple[str, list[int]]],
    id_to_entity: dict[str, Entity],
    output_name: str,
) -> DV8Clustering:
    """Convert deicide clustering to DV8-compatible format"""

    # Create unique names mapping
    entities = list(id_to_entity.values())
    unique_names = create_unique_entity_names(entities)

    root_structure: list[DV8ClusteringNode] = []

    def find_or_create_child(
        parent_nested: list[DV8ClusteringNode],
        cluster_id: int,
        level: int,
        parent_name: str = "",
    ) -> DV8ClusteringNode:
        """Find or create a child node in the nested structure"""

        # Build current node's name from parent's node name
        if parent_name == "":
            current_name = f"L{cluster_id}"
        else:
            prefix = "L" if level % 2 == 0 else "M"
            current_name = f"{parent_name}/{prefix}{cluster_id}"

        # return existing node if found
        for child in parent_nested:
            if child.type == "group" and child.name == current_name:
                return child

        # create new node if didn't find one with this name
        new_node = DV8ClusteringNode(name=current_name, type="group", nested=[])
        parent_nested.append(new_node)
        return new_node

    # Process each entry in deicide clustering output
    for hex_id, cluster_path in clustering:
        unique_entity_name = unique_names[hex_id]

        if not cluster_path:
            print(
                f"Warning: Empty cluster path for entity {unique_entity_name} ({hex_id}), skipping."
            )
            continue

        # Start at root
        current_nested = root_structure
        current_node = None
        parent_name = ""

        # Traverse through tree
        for level, cluster_id in enumerate(cluster_path):
            current_node = find_or_create_child(
                current_nested, cluster_id, level, parent_name # type: ignore
            )
            parent_name = current_node.name
            current_nested = current_node.nested

        # Add item node at the end of the path
        if current_node is not None:
            entity_item = DV8ClusteringNode(
                name=unique_entity_name,
                type="item",
            )
            # Append node to existing list
            current_node.nested.append(entity_item) # type: ignore

    # Add client entities as separate modules at root level for visualization in DV8
    for entity in id_to_entity.values():
        if entity.name.startswith("(Client)"):
            client_item = DV8ClusteringNode(
                name=entity.name,
                type="item",
            )
            root_structure.append(client_item)

    return DV8Clustering(
        schema_version="1.0", name=output_name, structure=root_structure
    )


def create_unique_entity_names(entities: list[Entity]) -> dict[str, str]:
    """Create unique names for entities, handling duplicates with (1), (2), etc."""
    name_counts: dict[str, int] = {}
    entity_to_unique_name: dict[str, str] = {}

    for entity in entities:
        base_name = entity.name

        if base_name not in name_counts:
            # First occurrence
            name_counts[base_name] = 1
            entity_to_unique_name[entity.id] = base_name
        else:
            # Duplicate
            count = name_counts[base_name]
            unique_name = f"{base_name} ({count})"
            name_counts[base_name] += 1
            entity_to_unique_name[entity.id] = unique_name

    return entity_to_unique_name


def create_dv8_dependency(
    id_to_entity: dict[str, Entity],
    internal_deps: list[Dep],
    client_deps: list[Dep],
    output_name: str,
) -> dict[str, Any]:
    """Create DV8-compatible dependency (DSM) data"""

    # Create ordered variables array (lexicographically)
    entities = list(id_to_entity.values())
    entities.sort(key=lambda e: e.name)
    variables = [entity.name for entity in entities]

    # Create entity name to index mapping
    entity_id_to_index = {entity.id: idx for idx, entity in enumerate(entities)}

    # Aggregate dependencies by type
    dep_matrix = aggregate_dependencies(internal_deps + client_deps, entity_id_to_index)

    # Convert to cells format
    cells = create_cells_from_matrix(dep_matrix)

    return {
        "@schemaVersion": "1.0",
        "name": output_name,
        "variables": variables,
        "cells": cells,
    }


def aggregate_dependencies(
    all_deps: list[Dep], entity_id_to_index: dict[str, int]
) -> dict[tuple[int, int], dict[str, float]]:
    """Aggregate dependencies by source, destination, and type"""

    dep_matrix: dict[tuple[int, int], dict[str, float]] = {}

    for dep in all_deps:
        # Skip if either entity not in our index
        if dep.src_id not in entity_id_to_index or dep.tgt_id not in entity_id_to_index:
            continue

        src_idx = entity_id_to_index[dep.src_id]
        tgt_idx = entity_id_to_index[dep.tgt_id]
        key = (src_idx, tgt_idx)

        if key not in dep_matrix:
            dep_matrix[key] = {}

        if dep.kind not in dep_matrix[key]:
            dep_matrix[key][dep.kind] = 0.0

        # Aggregate count
        dep_matrix[key][dep.kind] += 1.0

    return dep_matrix


def create_cells_from_matrix(
    dep_matrix: dict[tuple[int, int], dict[str, float]],
) -> list[dict[str, Any]]:
    """Convert dependency matrix to cells format"""

    cells: list[dict[str, Any]] = []
    for (src_idx, tgt_idx), type_counts in dep_matrix.items():
        cell: dict[str, Any] = {
            "src": src_idx,
            "dest": tgt_idx,
            "values": dict(type_counts),
        }
        cells.append(cell)

    return cells
