import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

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
parser.add_argument('-nc', '--num-of-clients', type=int, default=4, help='numer of clients')
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
	# adj = nx.adjacency_matrix(dataset.graph, np.sort(list(dataset.graph.nodes))).A
	# adj = torch.tensor(prerocess_adj(adj), dtype=torch.float32).to(dev)

	adj = None
	if os.path.exists('adj_matrix/{}_adj.npy'.format(dataset.datasetName)) is False:
		adj = nx.adjacency_matrix(dataset.graph, np.sort(list(dataset.graph.nodes))).A
		adj = torch.tensor(prerocess_adj(adj), dtype=torch.float32).to(dev)
		np.save('adj_matrix/{}_adj.npy'.format(dataset.datasetName), adj)
	else:
		adj = torch.tensor(np.load('adj_matrix/{}_adj.npy'.format(dataset.datasetName)), dtype=torch.float32).to(dev)

	# ------------------------------------- clients group -----------------------------------
	clients = ClientsGroup(dataset, args['num_of_clients'], args['labelNIID'], args['graphNIID'], dev)
	numOfClientsPerComm = int(max(args['num_of_clients'] * args['cFraction'], 1))
	numOfClientsPerComm = numOfClientsPerComm - 1  # remain the last one to be a new client

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

	# initialParameter
	initialParameter = {}
	for key, var in Net.state_dict().items():
		initialParameter[key] = var.clone()

	le = [1, 2, 5, 10, 20]
	for e in le:

		# ------------------------------------ FedAvg GCN train ----------------------------------
		print('epoch {}'.format(e))

		globalAcc = []
		for id, client in clients.clientsSet.items():
			client.testAcc = []

		globalParameters = {}
		for key, var in initialParameter.items():
			globalParameters[key] = var.clone()

		for i in range(args['num_comm']):
			order = np.random.permutation(len(clients.clientsSet) - 1)
			clientsIncomm = ['client{}'.format(k) for k in order[0: numOfClientsPerComm]]

			sumParameters = None
			for client in clientsIncomm:
				localParameters = clients.clientsSet[client].localUpdate(Net, lossFun, optimizer, globalParameters, e)

				if sumParameters is None:
					sumParameters = {}
					for key, var in localParameters.items():
						sumParameters[key] = var.clone()
				else:
					for var in sumParameters:
						sumParameters[var] = sumParameters[var] + localParameters[var]

			for var in globalParameters:
				globalParameters[var] = sumParameters[var] / numOfClientsPerComm

			for id, client in clients.clientsSet.items():
				localTestAcc = client.localTest(Net, globalParameters)

			with torch.no_grad():
				Net.load_state_dict(globalParameters, strict=True)
				Net.eval()
				preds = Net(features, adj)
				testPreds = torch.argmax(preds[testNodes], dim=1)
				testLabels = torch.argmax(labels[testNodes], dim=1)
				acc = (testPreds == testLabels).float().mean().item()
				globalAcc.append(acc)

		for id, client in clients.clientsSet.items():
			path = 'result/IID/local_epoch/{}/{}'.format(dataset.datasetName, id)
			test_mkdir(path)
			np.save(os.path.join(path, 'local_epoch_{}'.format(e)), client.testAcc)

		path = 'result/IID/local_epoch/{}/global'.format(dataset.datasetName)
		test_mkdir(path)
		np.save(os.path.join(path, 'local_epoch_{}'.format(e)), globalAcc)


def test_mkdir(path):
	if not os.path.isdir(path):
		os.mkdir(path)


if __name__ == "__main__":
	args = parser.parse_args()
	args = args.__dict__

	FedAvgGCNTrain(args)

	round = np.array([i + 1 for i in range(200)])

	path = r'result/IID/local_epoch/{}'.format(args['dataset'])
	dirs = ['client0', 'client1', 'client2', 'client3', 'global']
	le = [1, 2, 5, 10, 20]

	standard = {'cora': 0.8031, 'citeseer': 0.715, 'pubmed': 0.7893}

	for dir in dirs:
		acc = []
		path_ = os.path.join(path, dir)
		for e in le:
			file = os.path.join(path_, 'local_epoch_{}.npy'.format(e))
			data = np.load(file, allow_pickle=True)
			acc.append(data)

		plt.title('{}'.format(dir))
		plt.xlabel('communication round')
		plt.ylabel('Accuracy')

		plt.plot(round[0: 100], acc[0][0: 100], color='bisque', label='1')
		plt.plot(round[0: 100], acc[1][0: 100], color='orange', label='2')
		plt.plot(round[0: 100], acc[2][0: 100], color='lightcoral', label='5')
		plt.plot(round[0: 100], acc[3][0: 100], color='red', label='10')
		plt.plot(round[0: 100], acc[4][0: 100], color='maroon', label='20')
		plt.plot(round[0: 100], [standard[args['dataset']] for _ in range(200)][0: 100], color='green', label='standard', linestyle='dashed')
		plt.legend(loc='best')

		# plt.savefig('plt/{}_{}.png'.format(args['dataset'], dir))
		plt.show()
		plt.cla()
