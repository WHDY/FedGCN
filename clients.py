import numpy as np
import networkx as nx
import torch

from dataGenerator import dataGenerator
from utils import prerocess_adj


class Client:
	def __init__(self, features, adj, labels, trainNodesWithLabel, trainNods, testNodes, dev):
		self.features = torch.tensor(features, dtype=torch.float32).to(dev)
		self.labels = torch.tensor(labels, dtype=torch.float32).to(dev)
		self.adj = torch.tensor(adj, dtype=torch.float32).to(dev)
		self.trainNodesWithLabel = torch.tensor(trainNodesWithLabel, dtype=torch.long).to(dev)
		self.trainNodes = torch.tensor(trainNods, dtype=torch.long).to(dev)
		self.testNodes = torch.tensor(testNodes, dtype=torch.long).to(dev)

		self.testAcc = []

	def localUpdate(self, Net, lossFun, opti, globalParameters, localEpoch):
		Net.load_state_dict(globalParameters, strict=True)
		for epoch in range(localEpoch):
			Net.train()
			preds = Net(self.features, self.adj)
			loss = lossFun(preds[self.trainNodesWithLabel], torch.argmax(self.labels[self.trainNodesWithLabel], dim=1))
			loss.backward()
			opti.step()
			opti.zero_grad()

		return Net.state_dict()

	def localTest(self, Net, parameters):
		Net.load_state_dict(parameters, strict=True)
		with torch.no_grad():
			Net.eval()
			preds = Net(self.features, self.adj)
			testPreds = torch.argmax(preds[self.testNodes], dim=1)
			testLabels = torch.argmax(self.labels[self.testNodes], dim=1)
			acc = (testPreds == testLabels).float().mean().item()
			self.testAcc.append(acc)
		return self.testAcc[-1]


class ClientsGroup:
	def __init__(self, dataset, numOfClients, isLabeNonIID, isGraphStruNonIID, dev):
		self.dataset = dataset
		self.numOfClients = numOfClients
		self.isLableNonIID = isLabeNonIID
		self.isGraphStruNonIID = isGraphStruNonIID
		self.dev = dev

		self.clientsSet = {}

		self.datasetBalanceAllocation()

	def datasetBalanceAllocation(self):
		trainNodesWithLabelCopy = self.dataset.trainNodesWithLabel.copy()
		trainNodesCopy = self.dataset.trainNodes[len(trainNodesWithLabelCopy): ].copy()
		testNodesCopy = self.dataset.testNodes.copy()

		if (self.isLableNonIID | self.isGraphStruNonIID) is False:
			np.random.shuffle(trainNodesWithLabelCopy)
			np.random.shuffle(trainNodesCopy)
			np.random.shuffle(testNodesCopy)

			size1 = len(trainNodesWithLabelCopy) // self.numOfClients
			size2 = len(trainNodesCopy) // self.numOfClients
			size3 = len(testNodesCopy) // self.numOfClients
			for i in range(self.numOfClients):
				trainNodesWithLabel = trainNodesWithLabelCopy[i * size1: (i + 1) * size1]
				trainNodes = trainNodesCopy[i * size2: (i + 1) * size2]
				testNodes = testNodesCopy[i * size3: (i + 1) * size3]

				nodesReorder = np.sort(np.hstack((trainNodesWithLabel, trainNodes, testNodes)))
				features = self.dataset.features[nodesReorder]
				labels = self.dataset.labels[nodesReorder]
				adj = prerocess_adj(nx.adjacency_matrix(self.dataset.graph, nodesReorder).A)

				client = Client(features, adj, labels, np.arange(size1), np.arange(size1 + size2),
								np.arange(size1 + size2, size1 + size2 + size3), self.dev)
				self.clientsSet['client{}'.format(i)] = client

		elif (self.isLableNonIID & self.isGraphStruNonIID) is True:
			pass
		elif self.isLableNonIID is True:
			pass
		else:
			pass


if __name__ == "__main__":
	# a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
	# b = a.copy()
	# a[1] = [0, 0, 0]
	# a[0][0] = 0
	# print(a, b)

	# a = np.array([1, 2, 3])
	# b = np.array([4, 6, 5])
	# c = np.array([7, 9, 8])
	#
	# print(np.sort(np.hstack((a, b, c))))

	size1 = 5
	size2 = 15
	size3 = 20

	print(np.arange(size1))
	print(np.arange(size1 + size2))
	print(np.arange(size1 + size2, size1 + size2 + size3))

	datasetName = "cora"
	device = torch.device('cpu')

	dataset = dataGenerator(datasetName)
	clients = ClientsGroup(dataset, 10, False, False, device)

	print(1)


