import os
import argparse
from tqdm import tqdm

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F

from dataGenerator import dataGenerator
from utils import prerocess_adj
from Model import GCN
from clients import ClientsGroup


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg-GCN")
# -------------------------------------- device and dataset -------------------------------------------------
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-ds', '--dataset', type=str, default='cora', help='name of dataset')

# ------------------------------------ hyper-parameter of FedAvg --------------------------------------------
parser.add_argument('-nc', '--num-of-clients', type=int, default=1, help='numer of clients')
parser.add_argument('-cf', '--cFraction', type=float, default=1, help='0 means 1 client, 1 means total clients')
parser.add_argument('-ncomm', '--num-comm', type=int, default=200, help='number of communication rounds')
parser.add_argument('-E', '--local-epoch', type=int, default=1, help='local train epoch')
parser.add_argument('-lniid', '--labelNIID', type=bool, default=False, help='whether the label is Non-IID')
parser.add_argument('-gniid', '--graphNIID', type=bool, default=False, help='whether the graph structure is Non-IID')

# ------------------------------------- hyper-parameter of GCN ----------------------------------------------
parser.add_argument('-hd', "--hidden-layer-dim", type=int, default=16, help="size of hidden dimension")
parser.add_argument('-nGCL', "--num-of-GCLayer", type=int, default=2, help="number of graph convolutional layers")
parser.add_argument('-bias', "--bias", type=bool, default=False, help="whether to add bias")
parser.add_argument('-lr', "--learning-rate", type=float, default=0.01, help="learning rate")
parser.add_argument('-dp', "--dropout", type=float, default=0.5, help="dropout")


def FedAvgGCNTrain(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = torch.device("cpu")

    # dataset
    dataset = dataGenerator(args['dataset'])
    features = torch.tensor(dataset.features, dtype=torch.float32).to(dev)
    labels = torch.tensor(dataset.labels, dtype=torch.float32).to(dev)
    testNodes = torch.tensor(dataset.testNodes, dtype=torch.long).to(dev)

    adj = None
    if os.path.exists('adj_matrix/{}_adj.npy'.format(dataset.datasetName)) is False:
        adj = nx.adjacency_matrix(dataset.graph, np.sort(list(dataset.graph.nodes))).A
        adj = prerocess_adj(adj)
        np.save('adj_matrix/{}_adj.npy'.format(dataset.datasetName), adj)
        adj = torch.tensor(adj, dtype=torch.float32).to(dev)
    else:
        adj = torch.tensor(np.load('adj_matrix/{}_adj.npy'.format(dataset.datasetName)), dtype=torch.float32).to(dev)

    # ------------------------------------- clients group -----------------------------------
    clients = ClientsGroup(dataset, args['num_of_clients'], args['labelNIID'], args['graphNIID'], dev)
    numOfClientsPerComm = int(max(args['num_of_clients'] * args['cFraction'], 1))

    # --------------------------------------- GCN Model --------------------------------------
    Net = GCN(inDim=dataset.features.shape[1],
              hidDim=args['hidden_layer_dim'],
              outDim=dataset.labels.shape[1],
              numOfGCNLayers=args['num_of_GCLayer'],
              bias=args['bias'],
              dropout=args['dropout'])
    Net.to(dev)

    lossFun = F.nll_loss  # loss function
    optimizer = torch.optim.Adam(Net.parameters(), lr=args['learning_rate'], weight_decay=5e-4)  # adam optimizer

    # ------------------------------------ FedAvg GCN train ----------------------------------
    globalParameters = {}
    for key, var in Net.state_dict().items():
        globalParameters[key] = var.clone()

    for i in range(args['num_comm']):
        order = np.random.permutation(len(clients.clientsSet))
        clientsIncomm = ['client{}'.format(k) for k in order[0: numOfClientsPerComm]]

        sumParameters = None
        totalSize = 0
        for client in tqdm(clientsIncomm):
            localParameters, localTestSize = clients.clientsSet[client].localUpdate(Net, lossFun, optimizer, globalParameters, args['local_epoch'])

            if (i + 1) == args['num_comm']:
                localTestAcc = clients.clientsSet[client].localTest(Net, localParameters)
                print('{} model'.format(client))
                print('--local accuracy: {}%'.format(localTestAcc*100))

            if sumParameters is None:
                sumParameters = {}
                for key, var in localParameters.items():
                    sumParameters[key] = var.clone()*localTestSize
            else:
                for var in sumParameters:
                    sumParameters[var] = sumParameters[var] + localParameters[var]*localTestSize

            totalSize = totalSize + localTestSize

        for var in globalParameters:
            globalParameters[var] = sumParameters[var] / totalSize

        with torch.no_grad():
            Net.load_state_dict(globalParameters, strict=True)
            Net.eval()
            preds = Net(features, adj)
            testPreds = torch.argmax(preds[testNodes], dim=1)
            testLabels = torch.argmax(labels[testNodes], dim=1)
            acc = (testPreds == testLabels).float().mean().item()
            if (i + 1) == args['num_comm']:
                print('global model')
                print("--global accuracy： {}%".format(acc * 100))

    return acc


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    sum_acc = 0
    for i in range(10):
        sum_acc += FedAvgGCNTrain(args)

    print(sum_acc / 10)
