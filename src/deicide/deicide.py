import time
import logging

import numpy as np
import pandas as pd
from ordered_set import OrderedSet as oset

from deicide import ilp
from deicide.graph import group_by_scc, group_by_wcc, group_edges_by
from deicide.loading import Dataset
from deicide.naming import NameSimilarity


logging.basicConfig(
    filename="deicide.log",
    encoding="utf-8",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Text Options
TEXT_SIM_LOOKBACK = 1     # Set this "1" for normal bigrams
ALLOW_DUP_NAMES = True    # Should the sample space (for name similarity) be a set or multi-set?
TEXT_EDGE_MIN_SIM = 0.25  # Lower values take longer but use more available information
TEXT_EDGE_MULTIPLIER = 1  # Set to "0" to disable

# All edges weights are multiplied by this then rounded to the nearest integer
UNIT_EDGE_WEIGHT = 4096

# Balance paramater
CUT_EPS = 0.1

# When to stop splitting
WEIGHT_THRESH = 8 # change to 7


def group_sim(
    sim: NameSimilarity, a_names: list[str], b_names: list[str]
) -> list[float]:
    """Return a list of similarity scores between these two groups."""
    # Filenames are not included in our similarity comparison
    a_names = [n for n in a_names if sim.has_doc(n)]
    b_names = [n for n in b_names if sim.has_doc(n)]
    weights = []
    for a_name in a_names:
        for b_name in b_names:
            weights.append(sim.sim(a_name, b_name))
    return weights


def min_group_sim(sim: NameSimilarity, a_names: list[str], b_names: list[str]):
    """Return similarity score between the two groups using min-linkage."""
    weights = group_sim(sim, a_names, b_names)
    return np.min(weights) if len(weights) != 0 else 0


def avg_group_sim(sim: NameSimilarity, a_names: list[str], b_names: list[str]):
    """Return similarity score between the two groups using average-linkage."""
    weights = group_sim(sim, a_names, b_names)
    return np.average(weights) if len(weights) != 0 else 0


def max_group_sim(sim: NameSimilarity, a_names: list[str], b_names: list[str]):
    """Return similarity score between the two groups using max-linkage."""
    weights = group_sim(sim, a_names, b_names)
    return np.max(weights) if len(weights) != 0 else 0


def create_txt_edges(
    entities_df: pd.DataFrame, sim: NameSimilarity
) -> dict[tuple[int, int], float]:
    """Return a dict with text edges as keys and similarity scores as values. The edges
    are between `strong_id`s.
    """
    edges = {}
    nonfiles = entities_df[entities_df["kind"] != "file"]
    strong_names = nonfiles.groupby("strong_id")["name"].apply(list).to_dict()
    strong_ids = list(strong_names.keys())
    for a_ix in range(len(strong_ids)):
        a_names = strong_names[a_ix]
        for b_ix in range(a_ix + 1, len(strong_ids)):
            b_names = strong_names[b_ix]
            score = max_group_sim(sim, a_names, b_names)
            if score >= TEXT_EDGE_MIN_SIM:
                edges[(a_ix, b_ix)] = score
    return edges


# This function was extracted from a Jupyter notebook.
def cluster_dataset(ds: Dataset, *, use_threshold: bool) -> pd.DataFrame:
    # This dataframe contains both members of the god file and dependent and dependee files
    entities_df = ds.entities_df()
    edges = oset((r["src_id"], r["tgt_id"]) for _, r in ds.deps_df().iterrows())

    # Create an object for lexically comparing entity names
    similarity = NameSimilarity(
        list(ds.targets_df["name"]),
        allow_dup_names=ALLOW_DUP_NAMES,
        lookback=TEXT_SIM_LOOKBACK,
    )

    # Create a `name_id`` for each entity that groups targets according to their name
    entities_df["name_id"] = entities_df.groupby("name").ngroup()

    # Create a `strong_id`` for each entity that groups targets according the strongly
    # connected component of their name
    name_edges = group_edges_by(edges, entities_df["name_id"])
    entities_df["strong_id"] = group_by_scc(entities_df["name_id"], name_edges)

    # Create a `weak_id` for each entity that groups targets according the weakly
    # connected component of their strong_id
    strong_edges = group_edges_by(edges, entities_df["strong_id"])
    entities_df["weak_id"] = group_by_wcc(entities_df["strong_id"], strong_edges)

    # Create the text edges between `strong_id`s
    txt_edge_weights = create_txt_edges(entities_df, similarity)
    txt_edges = set(txt_edge_weights.keys())

    # Define some helper functions
    def get_entity_weight(id: int) -> int:
        """Return the weight of a single entity."""
        kind = entities_df.loc[id]["kind"]
        return 0 if kind == "file" else 1

    def get_strong_weight(strong_id: int) -> int:
        """Return the weight of a strongly connnected component."""
        ids = entities_df[entities_df["strong_id"] == strong_id].index
        return sum(get_entity_weight(id) for id in ids)

    def get_edge_weight(a_strong_id: int, b_strong_id: int) -> int:
        """Return the weight of an edge between two strongly connected components."""
        weight = 0
        key = (a_strong_id, b_strong_id)
        if key in strong_edges:
            weight += UNIT_EDGE_WEIGHT
        if key in txt_edges:
            txt_weight = txt_edge_weights[key] * UNIT_EDGE_WEIGHT
            weight += round(txt_weight * TEXT_EDGE_MULTIPLIER)
        return weight

    def cluster(
        dep_edges: set[tuple[int, int]],
        txt_edges: set[tuple[int, int]],
        active: set[int],
        cluster_name: str,
    ) -> dict[int, str]:
        """Recursively bisect the graph.

        Arguments:
        dep_edges -- Set of dependency edges between SCCs (indented to be ordered pairs)
        txt_edges -- Set of textual edges between SCCs (indented to be unordered pairs)
        active -- Set of currently active SCCs
        cluster_name -- Name of parent cluster

        Returns:
        A dict mapping `strong_id`s to a cluster name
        """
        def w(strong_id: int) -> int:
            if strong_id not in active:
                return 0
            return get_strong_weight(strong_id)

        # Print info
        n_active_entities = len([a for a in active if get_strong_weight(a) > 0])
        logging.info(f"[{cluster_name}] Starting... ({n_active_entities} entities)")
        logging.debug("======================================")
        active_nodes = {n: w(n) for n in active}
        active_dep_edges = {(u, v): get_edge_weight(u, v) for u, v in dep_edges if u in active and v in active}
        active_txt_edges = {(u, v): get_edge_weight(u, v) for u, v in txt_edges if u in active and v in active}
        logging.debug(sorted(list(active_nodes.items())))
        logging.debug(sorted(list(active_dep_edges.items())))
        logging.debug(sorted(list(active_txt_edges.items())))
        logging.debug("======================================")

        default_res = {i: cluster_name for i in active}

        if use_threshold:
            if sum(get_strong_weight(strong_id) for strong_id in active) <= WEIGHT_THRESH:
                logging.info("Aborted. Weight under threshold.")
                return default_res
        else:
            if n_active_entities < 2:
                logging.info("Aborted. Nothing left to cluster.")
                return default_res

        start = time.perf_counter()
        cut_weight, labels = ilp.partition2(
            dep_edges, txt_edges, w, get_edge_weight, 2, CUT_EPS, 10
        )
        if labels is None:
            logging.error("Aborted. Failed to partition.")
            return default_res
        elapsed = time.perf_counter() - start
        logging.info(f"Bisected with a cut weight of {cut_weight} in {elapsed:0.4f} secs.")

        active_A = active & {i for i, l in labels.items() if l == 0}
        active_B = active & {i for i, l in labels.items() if l == 1}
        res_A = cluster(dep_edges, txt_edges, active_A, cluster_name + ".A")
        res_B = cluster(dep_edges, txt_edges, active_B, cluster_name + ".B")
        return res_A | res_B

    # ...
    block_names = {}

    for weak_id in range(entities_df["weak_id"].max() + 1):
        # The strong_ids inside the current weakly connected component (wcc)
        wcc_nodes = set(entities_df[entities_df["weak_id"] == weak_id]["strong_id"])
        wcc_dep_edges = {
            (a, b) for a, b in strong_edges if a in wcc_nodes and b in wcc_nodes
        }
        wcc_txt_edges = {
            (a, b) for a, b in txt_edges if a in wcc_nodes and b in wcc_nodes
        }
        block_names |= cluster(
            wcc_dep_edges, wcc_txt_edges, wcc_nodes, cluster_name=f"R.W{weak_id}"
        )

    entities_df["block_name"] = [block_names.get(i) for i in entities_df["strong_id"]]
    entities_df["block_id"] = entities_df.groupby("block_name").ngroup()
    return entities_df
