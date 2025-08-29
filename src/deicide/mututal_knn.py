
from sklearn.base import defaultdict

def getMututalKnnGraph(base_graph: dict[tuple[int, int], float], k: int) -> dict[tuple[int, int], float]:
    # Compute the mutual k-NN graph from the base graph
    mutual_knn_graph: dict[tuple[int, int], float] = dict()
    # Get the k-NN for each node
    knn_graph: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (a, b), score in base_graph.items():
        knn_graph[a].append((b, score))
        knn_graph[b].append((a, score))
    # Keep only the top-k neighbors
    for node, neighbors in knn_graph.items():
        neighbors.sort(key=lambda x: x[1], reverse=True)
        real_k = min(k, len(neighbors))
        knn_graph[node] = neighbors[:real_k]
    # Create the mutual k-NN graph
    for node, neighbors in knn_graph.items():
        for n, score in neighbors:
            if node in {nn for nn, _ in knn_graph[n]}:
                mutual_knn_graph[(node, n)] = score
    return mutual_knn_graph
