import enum
from collections import Counter
from functools import cache

import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from deicide.algorithms import core
from deicide import naming
from deicide.loading import Dataset
from deicide.validation2 import Clustering


Doc = list[str]
Corpus = list[list[tuple[int, int]]]


def split(docs: list[Doc], ratio: float = 0.20) -> tuple[list[Doc], list[Doc]]:
    return train_test_split(docs, test_size=ratio, shuffle=True, random_state=0)  # type: ignore


def perplexity(lda: LdaModel, corpus: Corpus) -> float:
    return np.exp2(-lda.log_perplexity(corpus))  # type: ignore


class EntityKind(enum.StrEnum):
    FIELD = "field"
    METHOD = "method"


class Entity:
    def __init__(self, id: int, kind: EntityKind, name: str, tokens: list[str]):
        self._id = id
        self._kind = kind
        self._name = name
        self._tokens = tokens

    @property
    def id(self) -> int:
        return self._id

    @property
    def kind(self) -> EntityKind:
        return self._kind

    @property
    def name(self) -> str:
        return self._name

    @property
    def tokens(self) -> list[str]:
        return self._tokens


class GodClass:
    def __init__(self, entities: list[Entity], deps: Counter[tuple[int, int]]):
        self._entities = entities
        self._entities_by_id = {e.id: (i, e) for i, e in enumerate(entities)}
        if len(self._entities) != len(self._entities_by_id):
            raise ValueError("duplicate id")
        self._deps = deps

    @property
    def entities(self) -> list[Entity]:
        return self._entities

    @property
    def deps(self) -> Counter[tuple[int, int]]:
        return self._deps

    def to_entity_ix(self, id: int) -> int:
        entity_ix, _ = self._entities_by_id[id]
        return entity_ix

    @cache
    def methods(self) -> list[Entity]:
        return [e for e in self.entities if e.kind == EntityKind.METHOD]

    @cache
    def incoming(self, id: int) -> list[Entity]:
        return [self[u] for (u, v) in self.deps if v == id]

    @cache
    def outgoing(self, id: int) -> list[Entity]:
        return [self[v] for (u, v) in self.deps if u == id]

    @cache
    def usages(self, id: int) -> set[int]:
        return {e.id for e in self.outgoing(id) if e.kind == EntityKind.FIELD}

    def n_deps(self, u: int, v: int) -> int:
        return self.deps[(u, v)]

    @cache
    def n_deps_in(self, id: int) -> int:
        return sum(count for ((_, v), count) in self.deps.items() if v == id)

    @cache
    def ssm(self, u: int, v: int) -> float:
        return core.jaccard(self.usages(u), self.usages(v))

    @cache
    def directed_cdm(self, u: int, v: int) -> float:
        n_deps_in = self.n_deps_in(v)
        if n_deps_in == 0:
            return 0.0
        return self.n_deps(u, v) / n_deps_in

    @cache
    def cdm(self, u: int, v: int) -> float:
        return max(self.directed_cdm(u, v), self.directed_cdm(v, u))

    @cache
    def docs(self) -> list[Doc]:
        return [e.tokens for e in self.entities]

    @cache
    def vocab(self) -> Dictionary:
        return Dictionary(self.docs())

    def to_corpus(self, docs: list[Doc]) -> Corpus:
        vocab = self.vocab()
        return [vocab.doc2bow(doc) for doc in docs]  # type: ignore

    def find_best_k(self) -> int:
        # Insired by: http://freerangestats.info/blog/2017/01/05/topic-model-cv
        # Split documents and create cropus
        docs = self.docs()
        docs_train, docs_valid = split(docs)
        corpus_train = self.to_corpus(docs_train)
        corpus_valid = self.to_corpus(docs_valid)

        # Run LDA with different k's and find the best
        scores: dict[int, float] = {}
        for k in range(2, len(docs)):
            lda = LdaModel(corpus_train, id2word=self.vocab(), num_topics=k)
            scores[k] = perplexity(lda, corpus_valid)
        return min(scores, key=lambda k: scores[k])

    @cache
    def csm_mat(self) -> np.ndarray:
        # Run LDA
        k = self.find_best_k()
        docs = self.docs()
        corpus = self.to_corpus(docs)
        lda = LdaModel(corpus, id2word=self.vocab(), num_topics=k)

        # Create (n_samples, n_topics) matrix from LDA
        mat = np.zeros((len(docs), k))
        vocab = self.vocab()
        for i, doc in enumerate(docs):
            topic_dist = lda.get_document_topics(vocab.doc2bow(doc))
            for j, score in topic_dist:
                mat[(i, j)] = score

        # Calculate pairwise cosine similarities
        return cosine_similarity(mat)  # type: ignore

    @cache
    def csm(self, u: int, v: int) -> float:
        csm_mat = self.csm_mat()
        u_ix = self.to_entity_ix(u)
        v_ix = self.to_entity_ix(v)
        # Always access the upper-triangular portion
        return csm_mat[(min(u_ix, v_ix), max(u_ix, v_ix))]

    @cache
    def sim(self, u: int, v: int) -> float:
        return (1 / 3) * (self.ssm(u, v) + self.cdm(u, v) + self.csm(u, v))

    def dist(self, u: int, v: int) -> float:
        return 1.0 - self.sim(u, v)

    def __getitem__(self, id: int) -> Entity:
        _, entity = self._entities_by_id[id]
        return entity


class AkashDist(core.Dist):
    def __init__(self, god_class: GodClass):
        self._god_class = god_class

    def __call__(self, a: int, b: int) -> float:
        return self._god_class.dist(a, b)


def akash19(god_class: GodClass, *, shuffle: bool) -> Clustering:
    ids = {m.id for m in god_class.methods()}
    link_dist = core.AvgLinkDist(AkashDist(god_class))
    return core.hac(ids, link_dist, shuffle=shuffle).to_clustering()


# def many_akash19(
#     god_class: GodClass, n_trails: int, threshold_range: tuple[float, float]
# ) -> list[Clustering]:
#     thresh_min, thresh_max = threshold_range
#     clusterings: list[Clustering] = []
#     for _ in range(n_trails):
#         threshold = random.uniform(thresh_min, thresh_max)
#         clusterings.append(akash19(god_class, threshold, None))
#     return clusterings


def tokenize_identifiers(identifiers: str) -> list[str]:
    terms: list[str] = []
    for identifier in identifiers.split(","):
        terms.extend(naming.split_identifier(identifier))
    return preprocess_string(" ".join(terms))  # type: ignore


def to_godclass(ds: Dataset) -> GodClass:
    entities: list[Entity] = []
    for id, row in ds.targets_df.iterrows():
        if row["kind"] == "field":
            kind = EntityKind.FIELD
        elif row["kind"] == "method" or row["kind"] == "constructor":
            kind = EntityKind.METHOD
        else:
            continue
        tokens = tokenize_identifiers(row["identifiers"])
        entities.append(Entity(int(id), kind, row["name"], tokens))
    entity_ids = {e.id for e in entities}
    deps = {
        (int(r["src_id"]), int(r["tgt_id"])) for (_, r) in ds.target_deps_df.iterrows()
    }
    deps = {(u, v) for u, v in deps if u in entity_ids and v in entity_ids}
    return GodClass(entities, Counter(deps))
