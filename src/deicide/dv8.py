import json
from collections import defaultdict


def to_dsm(name, targets_df, deps_df) -> dict:
    ids = [id for id, _ in targets_df.iterrows()]
    vars = [row["name"] for _, row in targets_df.iterrows()]
    values = defaultdict(dict)
    for _, dep in deps_df.iterrows():
        src_id = dep["src_id"]
        tgt_id = dep["tgt_id"]
        kind = dep["kind"]
        values[(src_id, tgt_id)][kind] = 1.0
    cells = list()
    for ((src_id, tgt_id), v) in values.items():
         src_ix = ids.index(src_id)
         tgt_ix = ids.index(tgt_id)
         cells.append({"src": src_ix, "dest": tgt_ix, "values": v})
    dsm = {}
    dsm["schemaVersion"] = "1.0"
    dsm["name"] = name
    dsm["variables"] = vars
    dsm["cells"] = cells
    return dsm


def create_group(name: str):
    group = dict()
    group["@type"] = "group"
    group["name"] = name
    group["nested"] = list()
    return group


def create_item(name: str):
    item = dict()
    item["@type"] = "item"
    item["name"] = name
    return item


def to_inner_name(idx: int) -> str:
    if idx == 0:
        return "A"
    elif idx == 1:
        return "B"
    else:
        raise RuntimeError()


def add_to_root(root: list, idx_list: list[int], item_name: str):
    while idx_list[0] >= len(root):
        root.append(None)
    if root[idx_list[0]] is None:
        root[idx_list[0]] = create_group("W" + str(idx_list[0]))
    curr_group = root[idx_list[0]]
    for idx in idx_list[1:]:
        if len(curr_group["nested"]) == 0:
            curr_group["nested"] = [None, None]
        if curr_group["nested"][idx] is None:
            curr_group["nested"][idx] = create_group(to_inner_name(idx))
        curr_group = curr_group["nested"][idx]
    curr_group["nested"].append(create_item(item_name))


def to_idx_list(block_name: str) -> list[int]:
    idx_list = []
    weak_id_str = ""
    for c in reversed(block_name):
        if c == "A":
            idx_list.append(0)
        elif c == "B":
            idx_list.append(1)
        elif c in "0123456789":
            weak_id_str += c
    idx_list.append(int("".join(reversed(weak_id_str))))
    idx_list = list(reversed(idx_list))
    return idx_list


def to_drh(name, entities_df) -> dict:
    drh = dict()
    drh["@schemaVersion"] = "1.0"
    drh["name"] = name
    root = list()
    for _, row in entities_df.iterrows():
        add_to_root(root, to_idx_list(row["block_name"]), row["name"])
    drh["structure"] = root
    return drh


def write_dsm(filename, name, targets_df, deps_df):
    with open(filename, "w") as f:
        json.dump(to_dsm(name, targets_df, deps_df), f)


def write_drh(filename, name, entities_df):
    with open(filename, "w") as f:
        json.dump(to_drh(name, entities_df), f)