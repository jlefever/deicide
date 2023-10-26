import csv
from pathlib import Path
from functools import cache
from typing import NamedTuple
from collections import defaultdict

import pandas as pd


class JDeoRow(NamedTuple):
    filename: str
    name: str
    kind: str
    line: int
    cluster: str


@cache
def load_jdeo_project(jdeo_dir: str, project: str) -> list[JDeoRow]:
    clusters = dict()
    csv_file = Path(jdeo_dir, f"{project}.csv")
    if not csv_file.exists():
        return []
    with open(csv_file) as file:
        for row in csv.reader(file):
            filename, name, kind, line, cluster = row
            filename = filename.removeprefix("/")
            line = int(line)
            key = (filename, name, kind, line)
            if key not in clusters:
                clusters[key] = cluster
            elif cluster != clusters[key]:
                print(f"Entity `{key}` has inconsistent cluster identifiers")
    return [JDeoRow(*k, cluster=v) for k, v in clusters.items()]


@cache
def load_jdeo_subject(jdeo_dir: str, project: str, filename: str) -> list[JDeoRow]:
    rows = load_jdeo_project(jdeo_dir, project)
    return [r for r in rows if r.filename == filename]


def match_entities(
    jdeo_rows: list[JDeoRow], targets_df: pd.DataFrame
) -> dict[JDeoRow, int]:
    # Both our data and JDeo's data include line numbers. However, JDeo starts
    # counting at the doc comment while ours skips this and starts counting at
    # the signature. So instead of using the exact line number, we use the
    # order of the entities to match the records.
    targets_df = targets_df.copy()
    targets_df.sort_values(["start_row", "end_row"])

    id_map = defaultdict(list)

    for id, row in targets_df.iterrows():
        name = row["name"]
        # JDeo only has "field"s and "method"s
        kind = row["kind"] if row["kind"] != "constructor" else "method"
        # Each list is sorted in descending order by line number
        id_map[(name, kind)].insert(0, id)

    jdeo_id_map = dict()

    for row in sorted(jdeo_rows, key=lambda r: r.line):
        stack = id_map[(row.name, row.kind)]
        if len(stack) != 0:
            jdeo_id_map[row] = stack.pop()
        else:
            jdeo_id_map[row] = None

    return jdeo_id_map
