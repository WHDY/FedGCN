import os

import networkx as nx
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


class dataGenerator(object):
	"""
	ddtasetName: {"cora", "citeseer", "pubmed"}
	"""
	def __init__(self, datasetName):
		super(dataGenerator, self).__init__()

		self.datasetName = datasetName
		self.graph = None
		self.features = None
		self.labels = None
		self.trainNodes = None
		self.testNodes = None
		self.trainNodesWithLabel = None

		self.buildGraph(datasetName)

	def buildGraph(self, datasetName):
		filePath = os.path.join("dataWithGCNSetting", datasetName)  # data path

		# ------------- construct graph ------------------------------------------------
		with open(os.path.join(filePath, "ind.{}.graph".format(datasetName)), 'rb') as f:
			self.graph = nx.from_dict_of_lists(pkl.load(f, encoding='latin1'))
			# print(type(self.graph))
			# print(datasetName + ":")
			# print("--number of nodes: {}".format(nx.number_of_nodes(self.graph)))
			# print("--number of edges: {}".format(nx.number_of_edges(self.graph)))
			# print("--number of isolated subgraph: {}".format(nx.number_connected_components(self.graph)))

		# ------------- read features and labels ----------------------------------------
		suffixes = ["x", "tx", "allx", "y", "ty", "ally"]
		contents = []
		for i, suffix in enumerate(suffixes):
			path = os.path.join(filePath, "ind.{}.{}".format(datasetName, suffix))
			with open(path, 'rb') as f:
				if i < 3:
					contents.append(pkl.load(f, encoding='latin1').A)  # 将 scipy.sparse.csr.csr_matrix 转为 numpy.ndarray
				else:
					contents.append(pkl.load(f, encoding='latin1'))

		x, tx, allx, y, ty, ally = tuple(contents)

		# ---------- read index of testing nodes ----------------------------------------
		testIdx = []
		with open(os.path.join(filePath, "ind.{}.test.index".format(datasetName)), 'rb') as f:
			for line in f:
				testIdx.append(int(line.strip()))
		testIdx = np.array(testIdx)
		testIdxReorder = np.sort(testIdx)

		# ---------- Fix citeseer dataset -----------------------------------------------
		if datasetName == "citeseer":
			testIdxFull = range(min(testIdx), max(testIdx) + 1)
			txExtended = np.zeros(shape=(len(testIdxFull), x.shape[1]))
			txExtended[testIdxReorder - min(testIdx), :] = tx
			tx = txExtended
			tyExtended = np.zeros((len(testIdxFull), y.shape[1]))
			tyExtended[testIdxReorder - min(testIdx), :] = ty
			ty = tyExtended

		# ---------- feature process ---------------------------------------------------
		self.features = np.vstack((allx, tx))
		self.features[testIdx, :] = self.features[testIdxReorder, :]
		rowSum = np.array(self.features.sum(1))
		a = np.power(rowSum, -1).flatten()
		a[np.isinf(a)] = 0.0
		aMat = np.diag(a)
		self.features = aMat.dot(self.features)

		# ---------- label process ---------------------------------------------------
		self.labels = np.vstack((ally, ty))
		self.labels[testIdx, :] = self.labels[testIdxReorder, :]

		self.testNodes = np.sort(testIdx)
		self.trainNodes = np.arange(min(testIdx))
		self.trainNodesWithLabel = np.arange(len(x))


if __name__ == "__main__":
	cora = dataGenerator("pubmed")
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

	graph.add_node(6)
	graph.add_node(8)
	graph.add_edge(6, 8)

	print(nx.number_connected_components(graph))

	graph = graph.subgraph([1,2,4,5])
	nx.draw(graph, with_labels=True, font_weight='bold')
	# nx.draw_shell(graph)
	plt.show()
