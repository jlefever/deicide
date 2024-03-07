from collections import defaultdict
from dataclasses import dataclass
from typing import Any
import json

from deicide.validation2 import Clustering, ClusterPath


@dataclass
class Leaf:
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {"@type": "item", "name": self.name}


@dataclass
class Branch:
    name: str
    children: "list[Branch | Leaf]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "@type": "group",
            "name": self.name,
            "nested": [c.to_dict() for c in self.children],
        }


def fix_name(name: str) -> str:
    return name.split("/")[-1]


def build_roots(clustering: Clustering, names: dict[int, str]) -> list[Branch]:
    roots: list[Branch] = []
    branches: dict[ClusterPath, Branch] = {}

    def get_branch(cluster: ClusterPath) -> Branch:
        if cluster in branches:
            return branches[cluster]
        branch = Branch("/".join(cluster.segments), [])
        branches[cluster] = branch
        parent = cluster.parent()
        if parent is None:
            roots.append(branch)
        else:
            get_branch(parent).children.append(branch)
        return branch

    for id, cluster in clustering.pairs:
        get_branch(cluster).children.append(Leaf(fix_name(names[id])))

    for branch in branches.values():
        branch.children.sort(key=lambda x: x.name, reverse=True)
    roots.sort(key=lambda x: x.name, reverse=True)
    return roots


def to_dv8_clx(name: str, clustering: Clustering, names: dict[int, str]) -> str:
    clx = {
        "@schemaVersion": "1.0",
        "name": name,
        "structure": [r.to_dict() for r in build_roots(clustering, names)],
    }
    return json.dumps(clx, indent=4)


def to_dv8_dsm(name: str, entities_df: Any, deps_df: Any) -> str:
    ids = [id for id, _ in entities_df.iterrows()]
    vars = [fix_name(row["name"]) for _, row in entities_df.iterrows()]
    values: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for _, dep in deps_df.iterrows():
        src_id = int(dep["src_id"])
        tgt_id = int(dep["tgt_id"])
        kind = str(dep["kind"])
        values[(src_id, tgt_id)][kind] = 1.0
    cells: list[dict[str, Any]] = list()
    for (src_id, tgt_id), v in values.items():
        src_ix = ids.index(src_id)
        tgt_ix = ids.index(tgt_id)
        cells.append({"src": src_ix, "dest": tgt_ix, "values": v})
    dsm = {}
    dsm["schemaVersion"] = "1.0"
    dsm["name"] = name
    dsm["variables"] = vars
    dsm["cells"] = cells
    return json.dumps(dsm, indent=4)


# def foo() -> dict[str, Any]:
#     a = Leaf("a", 0)
#     b = Leaf("b", 1)
#     c = Branch("c", [a, b])
#     d = Branch("d", [c])
#     return asdict(d)
