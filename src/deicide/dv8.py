from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from deicide.core import Entity

@dataclass
class DV8ClusteringNode:
    """Represents a node in the DV8 clustering tree."""
    name: str
    type: Literal["group", "item"]
    nested: list[DV8ClusteringNode] | None = None

    def __post_init__(self):
        """Ensure that "item" nodes do not have nested children"""
        if self.type == "item" and self.nested is not None:
            raise ValueError("Item nodes cannot have nested children")

    def to_dict(self) -> dict[str, str | list]:
        """Convert to dictionary for JSON serialization"""
        result: dict[str, str | list] = {
            "@type": self.type,
            "name": self.name
        }
        if self.type == "group" and self.nested is not None:
            result["nested"] = [node.to_dict() for node in self.nested]
        return result

@dataclass
class DV8Clustering:
    """Root data structure for DV8 clustering format"""
    schema_version: str = "1.0" # set default to 1.0
    name: str = ""
    structure: list[DV8ClusteringNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, str | list]:
        """Convert to dictionary for JSON serialization"""
        return {
            "@schemaVersion": self.schema_version,
            "name": self.name,
            "structure": [node.to_dict() for node in self.structure]
        }
    
def create_dv8_clustering(clustering: list[tuple[str, list[int]]], id_to_entity: dict[str, Entity], output_name: str) -> DV8Clustering:
    """Convert deicide clustering to DV8-compatible format"""

    # Create unique names mapping
    entities = list(id_to_entity.values())
    unique_names = create_unique_entity_names(entities)

    root_structure: list[DV8ClusteringNode] = []

    def find_or_create_child(
        parent_nested: list[DV8ClusteringNode],
        cluster_id: int,
        level: int,
        parent_name: str = ""
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
        new_node = DV8ClusteringNode(
            name=current_name,
            type="group",
            nested=[]
        )
        parent_nested.append(new_node)
        return new_node

    # Process each entry in deicide clustering output
    for hex_id, cluster_path in clustering:
        unique_entity_name = unique_names[hex_id]

        if not cluster_path:
            print(f"Warning: Empty cluster path for entity {unique_entity_name} ({hex_id}), skipping.")
            continue
        
        # Start at root
        current_nested = root_structure
        current_node = None
        parent_name = ""

        # Traverse through tree
        for level, cluster_id in enumerate(cluster_path):
            current_node = find_or_create_child(current_nested, cluster_id, level, parent_name)
            parent_name = current_node.name
            current_nested = current_node.nested
        
        # Add item node at the end of the path
        if current_node is not None:
            entity_item = DV8ClusteringNode(
                name=unique_entity_name,
                type="item",
            )
            # Append node to existing list
            current_node.nested.append(entity_item)
        
    return DV8Clustering(
        schema_version="1.0",
        name=output_name,
        structure=root_structure
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