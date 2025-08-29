import abc

from deicide.kielaclark import KielaClark
from deicide.lsi import LSI
from deicide.corpus import tokenize_corpus

KIELA_CLARK_MIN_SIM = 0.25
LSI_MIN_SIM = 0.25


class SemanticSimilarity(abc.ABC):
    @abc.abstractmethod
    def fit(self, corpus: dict[str, str]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def sim(self, a_id: str, b_id: str) -> float:
        raise NotImplementedError


class KielaClarkSimilarity(SemanticSimilarity):
    def __init__(self) -> None:
        self._corpus: dict[str, str] = dict()

    def fit(self, corpus: dict[str, str]) -> None:
        self._corpus = corpus
        self._kielaclark = KielaClark(list(corpus.values()))

    def sim(self, a_id: str, b_id: str) -> float:
        a_doc, b_doc = self._corpus[a_id], self._corpus[b_id]
        score = self._kielaclark.sim(a_doc, b_doc)
        if score < KIELA_CLARK_MIN_SIM:
            return 0.0
        return score


class LSISimilarity(SemanticSimilarity):
    def __init__(self, cutoff_method: str) -> None:
        self._corpus: dict[str, str] = dict()
        self._cutoff_method = cutoff_method

    def fit(self, corpus: dict[str, str]) -> None:
        self._corpus = corpus
        self._lsi: LSI = LSI(tokenize_corpus(corpus))

    def sim(self, a_id: str, b_id: str) -> float:
        # a_doc, b_doc = self._corpus[a_id], self._corpus[b_id]
        score: float = self._lsi.sim(a_id, b_id)
        if (self._cutoff_method == "threshold"):
            if score < LSI_MIN_SIM:
                return 0.0
        return score
