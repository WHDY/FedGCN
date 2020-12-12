import random
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

        return Net.state_dict(), len(self.trainNodesWithLabel)

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
    def __init__(self, dataset, numOfClients, isLabelNonIID, isGraphStruNonIID, dev):
        self.dataset = dataset
        self.numOfClients = numOfClients
        self.dev = dev

        self.clientsSet = {}

        if (isLabelNonIID | isGraphStruNonIID) is False:
            self.datasetIIDAllocation()
        elif isLabelNonIID is True and isGraphStruNonIID is False:
            self.datasetLableNonIIDAllocation()
        elif isLabelNonIID is False and isGraphStruNonIID is True:
            self.datasetGtNonIIDAllocation()
        else:
            pass

    def datasetIIDAllocation(self):
        trainNodesWithLabelCopy = self.dataset.trainNodesWithLabel.copy()
        trainNodesCopy = self.dataset.trainNodes[len(trainNodesWithLabelCopy):].copy()
        testNodesCopy = self.dataset.testNodes.copy()

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

    def datasetLableNonIIDAllocation(self):
        '''
        fix 3 clients, extremely label Non-IID
        '''
        labels = np.argmax(self.dataset.labels, axis=1)
        dataSplit = []
        for label in range(self.dataset.labels.shape[1]):
            dataSplit.append(np.where(labels == label)[0])

        random.shuffle(dataSplit)
        size = self.dataset.labels.shape[1] // 3
        standardLine = {'cora': [140, 1708], 'citeseer': [120, 2327], 'pubmed': [60, 18717]}
        for i in range(3):
            if i != 2:
                nodes = np.hstack((dataSplit[i * size: (i + 1) * size]))
            else:
                nodes = np.hstack((dataSplit[i * size:]))

            trainNodesWithLabel = nodes[np.where(nodes < standardLine[self.dataset.datasetName][0])]
            trainNodes = nodes[np.where(nodes < standardLine[self.dataset.datasetName][1])]
            testNodes = nodes[np.where(nodes >= standardLine[self.dataset.datasetName][1])]

            nodesReorder = np.sort(np.hstack((trainNodes, testNodes)))
            features = self.dataset.features[nodesReorder]
            labels = self.dataset.labels[nodesReorder]
            adj = prerocess_adj(nx.adjacency_matrix(self.dataset.graph, nodesReorder).A)

            client = Client(features, adj, labels, np.arange(len(trainNodesWithLabel)), np.arange(len(trainNodes)),
                            np.arange(len(trainNodes), len(trainNodes) + len(testNodes)), self.dev)

            self.clientsSet['client{}'.format(i)] = client

    def datasetGtNonIIDAllocation(self):
        degreeList = nx.degree(self.dataset.graph, np.sort(list(self.dataset.graph.nodes)))
        standardLine = {'cora': [2, 4, 140, 1708], 'citeseer': [1, 2, 120, 2327], 'pubmed': [1, 3, 60, 18717]}
        dataSplit = [[], [], []]
        for id, degree in degreeList:
            if degree <= standardLine[self.dataset.datasetName][0]:
                dataSplit[0].append(id)
            elif standardLine[self.dataset.datasetName][0] < degree <= standardLine[self.dataset.datasetName][1]:
                dataSplit[1].append(id)
            else:
                dataSplit[2].append(id)

        for i in range(3):
            nodes = np.array(dataSplit[i])
            trainNodesWithLabel = nodes[np.where(nodes < standardLine[self.dataset.datasetName][2])]
            trainNodes = nodes[np.where(nodes < standardLine[self.dataset.datasetName][3])]
            testNodes = nodes[np.where(nodes >= standardLine[self.dataset.datasetName][3])]

            nodesReorder = np.sort(np.hstack((trainNodes, testNodes)))
            features = self.dataset.features[nodesReorder]
            labels = self.dataset.labels[nodesReorder]
            adj = prerocess_adj(nx.adjacency_matrix(self.dataset.graph, nodesReorder).A)

            client = Client(features, adj, labels, np.arange(len(trainNodesWithLabel)), np.arange(len(trainNodes)),
                            np.arange(len(trainNodes), len(trainNodes) + len(testNodes)), self.dev)

            self.clientsSet['client{}'.format(i)] = client


if __name__ == "__main__":
    # a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    # b = a.copy()
    # a[1] = [0, 0, 0]
    # a[0][0] = 0
    # print(a, b)

    a = np.array([1, 2, 3])
    b = np.array([4, 6, 5])
    c = np.array([7, 9, 8])
    x = [a, b, c]

    print(x[0:2])
    print(np.sort(np.hstack((x[0:2]))))

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

    x = [[1, 2, 3], [1, 2], [0, 1, 3]]
    random.shuffle(x)
    print(x)
