import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from sklearn.decomposition import TruncatedSVD  # type: ignore

TOP_K_SINGULAR_VALUES = 100


class LSI:
    def __init__(self, corpus: dict[str, list[str]]) -> None:
        self._corpus = corpus
        self._id_to_ix = {id: ix for ix, id in enumerate(sorted(corpus.keys()))}
        self._ix_to_id = {ix: id for id, ix in self._id_to_ix.items()}
        self._terms: list[str] = sorted({t for doc in corpus.values() for t in doc})

        start_time = time.time()
        num_docs = len(self._corpus)
        num_terms = len(self._terms)
        # print(f"Number of documents: {num_docs}")
        # print(f"Number of terms: {num_terms}")


        A = np.zeros((len(self._terms), len(self._corpus)))
        self._A = A
        for id, terms in self._corpus.items():
            for term in terms:
                term_index = self._terms.index(term)
                A[term_index, self._id_to_ix[id]] = 1

        k = min(TOP_K_SINGULAR_VALUES, len(self._terms), len(self._corpus))
        V = TruncatedSVD(n_components=k, random_state=42).fit_transform(A.T)
        print(f"VT shape: {np.shape(V)}")
        # print(f"U: {np.shape(U)}, S: {np.shape(S)}, VT: {np.shape(VT)}")

        self._ix_to_vec: dict[int, np.ndarray] = {
            ix: vec[:k].copy() for ix, vec in enumerate(V)
        }
        # At the end of __init__, after all processing:
        end_time = time.time()
        run_time = end_time - start_time
        with open("lsi_stats.txt", "a") as f:
            f.write(f"Documents: {num_docs}, Terms: {num_terms}, Run time: {run_time:.4f} seconds\n")

    # Assuming that doc exists in built tf-idf doc matrix
    def sim(self, a_doc_id: str, b_doc_id: str) -> float:
        a_ix = self._id_to_ix[a_doc_id]
        b_ix = self._id_to_ix[b_doc_id]
        print(f"a_id: {a_doc_id}, b_id: {b_doc_id}")
        print(f"a_ix: {a_ix}, b_ix: {b_ix}")
        a_vec = self._ix_to_vec[a_ix]
        b_vec = self._ix_to_vec[b_ix]

        a_vec = a_vec / np.linalg.norm(a_vec) if np.linalg.norm(a_vec) != 0 else a_vec
        b_vec = b_vec / np.linalg.norm(b_vec) if np.linalg.norm(b_vec) != 0 else b_vec

        return float(a_vec.dot(b_vec))
