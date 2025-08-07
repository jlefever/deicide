from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

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