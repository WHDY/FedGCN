import os

import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F

from dataGenerator import dataGenerator
from utils import prerocess_adj
from Model import GCN


def train(dataset, dev):
    trainNodes = torch.tensor(dataset.trainNodes, dtype=torch.long).to(dev)
    trainNodesWithLabel  =torch.tensor(dataset.trainNodesWithLabel, dtype=torch.long).to(dev)
    testNodes = torch.tensor(dataset.testNodes, dtype=torch.long).to(dev)
    features = torch.tensor(dataset.features, dtype=torch.float32).to(dev)
    labels = torch.tensor(dataset.labels, dtype=torch.float32).to(dev)

    adj = None
    if os.path.exists('adj_matrix/{}_adj.npy'.format(dataset.datasetName)) is False:
        adj = nx.adjacency_matrix(dataset.graph, np.sort(list(dataset.graph.nodes))).A
        adj = torch.tensor(prerocess_adj(adj), dtype=torch.float32).to(dev)
        np.save('adj_matrix/{}_adj.npy'.format(dataset.datasetName), adj)
    else:
        adj = torch.tensor(np.load('adj_matrix/{}_adj.npy'.format(dataset.datasetName)), dtype=torch.float32).to(dev)

    Net = GCN(inDim=dataset.features.shape[1],
              hidDim=16,
              outDim=dataset.labels.shape[1],
              numOfGCNLayers=2,
              bias=False,
              dropout=0.5)
    Net.to(dev)

    optimizer = torch.optim.Adam(Net.parameters(), lr=0.01, weight_decay=5e-4)  # adam optimizer

    # ------------------------------------------ train ------------------------------------------ #
    acc = 0.0
    for epoch in range(200):
        Net.train()
        optimizer.zero_grad()
        out = Net(features, adj)
        loss = F.nll_loss(out[trainNodesWithLabel], torch.argmax(labels[trainNodesWithLabel], dim=1))
        loss.backward()
        # print(epoch, loss.item())
        optimizer.step()

        with torch.no_grad():
            Net.eval()
            preds = Net(features, adj)
            testPreds = torch.argmax(preds[testNodes], dim=1)
            testLabels = torch.argmax(labels[testNodes], dim=1)
            acc = (testPreds == testLabels).float().mean().item()
            # print(epoch, acc)

    print(acc)
    return acc


if __name__ == "__main__":
    device = torch.device('cpu')

    datasetName = "cora"
    dataset = dataGenerator(datasetName)

    # train(dataset, device)

    # a = torch.tensor([1, 2, 3, 4, 5, 6])
    # idx = torch.tensor([1, 2])
    # print(a[idx])

    sumAcc = 0
    for i in range(10):
        sumAcc = sumAcc + train(dataset, device)
    print('{}: {}%'.format(datasetName, sumAcc*10))

