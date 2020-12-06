import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    D = np.power(rowsum, -0.5).flatten()
    D[np.isinf(D)] = 0.0
    DMat = np.diag(D)
    return adj.dot(DMat).transpose().dot(DMat)


def prerocess_adj(adj):
    adjAddSelfLoop = adj + np.eye(adj.shape[0])
    adjNormalized = normalize_adj(adjAddSelfLoop)
    return adjNormalized


if __name__ == "__main__":
    graph = nx.Graph()
    graph.add_node(1)
    graph.add_node(2)
    graph.add_node(7)
    graph.add_node(4)
    graph.add_node(5)

    graph.add_edge(1, 2)
    graph.add_edge(1, 4)
    graph.add_edge(1, 7)
    graph.add_edge(1, 5)
    graph.add_edge(4, 7)
    graph.add_edge(4, 5)

    adj = nx.adjacency_matrix(graph, np.sort(graph.nodes)).A
    print(list(graph.nodes))
    print(adj)

    print(prerocess_adj(adj))

