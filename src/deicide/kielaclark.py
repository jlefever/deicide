import math
from itertools import chain, pairwise
from functools import cache
from collections import Counter

import numpy as np
from ordered_set import OrderedSet as oset
from nltk.stem import PorterStemmer  # type: ignore


class KielaClark:
    def __init__(self, names: list[str], lookback: int = 1) -> None:
        # Populate a counter for term-document pairs (aka occurrences)
        pair_counts = Counter(chain(*(_to_occurrences(n, lookback) for n in names)))

        # Populate counters for documents (aka names or identifiers) and terms
        term_counts = Counter(t for t, _ in pair_counts.elements())
        doc_counts = Counter(d for _, d in pair_counts.elements())

        # Remove isolated terms from vocabulary
        isolated_terms = {t for t, c in term_counts.items() if c <= 1}
        pair_counts = Counter(
            {(t, d): c for (t, d), c in pair_counts.items() if t not in isolated_terms}
        )
        term_counts = Counter(t for t, _ in pair_counts.elements())
        doc_counts = Counter(d for _, d in pair_counts.elements())

        # Get total
        total = pair_counts.total()

        # Define functions for the probabilities and mutual information
        @cache
        def p_i_1(term: str) -> float:
            "Evaluates P(X_i = 1) where i is the term."
            return term_counts[term] / total

        @cache
        def p_i_0(term: str) -> float:
            "Evaluates P(X_i = 0) where i is the term."
            return (total - term_counts[term]) / total

        @cache
        def p_j_1(doc: str) -> float:
            "Evaluates P(Y_j = 1) where j is the document."
            return doc_counts[doc] / total

        @cache
        def p_j_0(doc: str) -> float:
            "Evaluates P(Y_j = 0) where j is the document."
            return (total - doc_counts[doc]) / total

        @cache
        def p_ij_11(term: str, doc: str) -> float:
            "Evaluates P(X_i = 1; Y_j = 1) where i is the term and j is the document."
            return pair_counts[term, doc] / total

        @cache
        def p_ij_10(term: str, doc: str) -> float:
            "Evaluates P(X_i = 1; Y_j = 0) where i is the term and j is the document."
            return (term_counts[term] - pair_counts[term, doc]) / total

        @cache
        def p_ij_01(term: str, doc: str) -> float:
            "Evaluates P(X_i = 0; Y_j = 1) where i is the term and j is the document."
            return (doc_counts[doc] - pair_counts[term, doc]) / total

        @cache
        def p_ij_00(term: str, doc: str) -> float:
            "Evaluates P(X_i = 0; Y_j = 0) where i is the term and j is the document."
            return (
                total + pair_counts[term, doc] - term_counts[term] - doc_counts[doc]
            ) / total

        def log(x: float) -> float:
            return 0.0 if x == 0.0 else math.log(x)

        def mi(term: str, doc: str) -> float:
            "Evaluates mutual information I(X_i; Y_j) where i is the term and j is the document."
            a = p_ij_11(term, doc) * log(
                p_ij_11(term, doc) / (p_i_1(term) * p_j_1(doc))
            )
            b = p_ij_10(term, doc) * log(
                p_ij_10(term, doc) / (p_i_1(term) * p_j_0(doc))
            )
            c = p_ij_01(term, doc) * log(
                p_ij_01(term, doc) / (p_i_0(term) * p_j_1(doc))
            )
            d = p_ij_00(term, doc) * log(
                p_ij_00(term, doc) / (p_i_0(term) * p_j_0(doc))
            )
            return a + b + c + d

        # Create ordered sets for the terms and docs to use as the canonical ordering
        self.terms = oset(term_counts)
        self.docs = oset(doc_counts)

        # Create a rectangular matrix to record I(X_i; Y_j) values
        arr = np.zeros((len(self.terms), len(self.docs)))
        for i, term in enumerate(self.terms):
            for j, doc in enumerate(self.docs):
                arr[i, j] = mi(term, doc)

        # Define positive correlation
        def center(vec: np.ndarray) -> np.ndarray:
            return vec - np.mean(vec)

        def norm(vec: np.ndarray) -> float:
            return float(np.linalg.norm(vec))

        def pos_cor(a: np.ndarray, b: np.ndarray) -> float:
            return max(0, np.dot(center(a), center(b)) / (norm(a) * norm(b)))

        # Create a square matrix to record correlation values
        self.sim_mat = np.zeros((len(self.docs), len(self.docs)))
        for i in range(len(self.docs)):
            for j in range(i, len(self.docs)):
                self.sim_mat[i, j] = self.sim_mat[j, i] = pos_cor(arr[:, i], arr[:, j])

        # Create a square dist
        self.dist_mat = 1 - self.sim_mat

    def has_doc(self, doc: str) -> bool:
        return _normalize_name(doc) in self.docs

    def get_doc_ix(self, doc: str) -> int:
        return self.docs.index(_normalize_name(doc))  # type: ignore

    def sim(self, a_doc: str, b_doc: str) -> float:
        if not self.has_doc(a_doc) or not self.has_doc(b_doc):
            return 0.0
        return float(self.sim_mat[self.get_doc_ix(a_doc), self.get_doc_ix(b_doc)])

    def most_sim(self, doc: str, n: int) -> list[tuple[str, float]]:
        doc_ix = self.get_doc_ix(doc)
        most_sim_indices = reversed(np.argsort(self.sim_mat[doc_ix])[-n:-1])
        return [(self.docs[ix], float(self.sim_mat[doc_ix, ix])) for ix in most_sim_indices]


def _to_occurrences(name: str, lookback: int) -> list[tuple[str, str]]:
    normalized = _normalize_name(name)
    terms = _termize(name)
    bigrams = _bigramize(terms, lookback)
    return [(t, normalized) for t in terms + bigrams]


@cache
def _normalize_name(name: str) -> str:
    return "_".join(_termize(name))


@cache
def _termize(name: str) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(z) for z in _split_identifier(name)]  # type: ignore


def _bigramize(terms: list[str], lookback: int) -> list[str]:
    bigrams: list[str] = []
    for i, curr in enumerate(terms):
        for prev in terms[max(i - lookback, 0) : i]:
            bigrams.append(f"{prev}-{curr}")
    return bigrams


def _split_identifier(name: str) -> list[str]:
    by_spaces = name.split(" ")
    by_underscores = chain(*(z.split("_") for z in by_spaces))
    return list(chain(*(_split_camel(z) for z in by_underscores)))


def _split_camel(name: str) -> list[str]:
    if name.isupper():
        return [name.lower()]
    indices = [i for i, x in enumerate(name) if x.isupper() or x.isnumeric()]
    indices = [0] + indices + [len(name)]
    return _join_singles([name[a:b].lower() for a, b in pairwise(indices)])


def _join_singles(terms: list[str]) -> list[str]:
    ret: list[str] = []
    joined_term: list[str] = []
    for t in terms:
        if len(t) == 1:
            joined_term.append(t[0])
        elif len(t) > 1:
            if len(joined_term) > 0:
                ret.append("".join(joined_term))
                joined_term = []
            ret.append(t)
    if len(joined_term) > 0:
        ret.append("".join(joined_term))
    return ret
