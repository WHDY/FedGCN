import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class GraphConvolution(nn.Module):
	def __init__(self, inDim, outDim, bias=False, dropout=0.0):
		super(GraphConvolution, self).__init__()

		self.weight = nn.Parameter(torch.Tensor(inDim, outDim))
		if bias:
			self.bias = nn.Parameter(torch.Tensor(outDim))
		else:
			self.register_parameter('bias', None)

		self.reset_parameters()

		self.dropout = nn.Dropout(dropout)

	def reset_parameters(self):
		# stdv = 1. / np.sqrt(self.weight.size(1))
		# self.weight.data.uniform_(-stdv, stdv)
		stdv = np.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
		self.weight.data.uniform_(-stdv, stdv)
		pass
		if self.bias is not None:
			# self.bias.data.uniform_(-stdv, stdv)
			self.bias.data.fill_(0)

	def forward(self, input, adj):
		tensor = torch.matmul(input, self.weight)
		output = torch.matmul(adj, tensor)
		if self.bias is not None:
			output = output + self.bias

		output = self.dropout(F.relu(output))

		return output


class GCN(nn.Module):
	def __init__(self, inDim, outDim, hidDim, numOfGCNLayers, bias=False, dropout=0.0):
		super(GCN, self).__init__()

		self.GCLayers = nn.ModuleList()
		self.GCLayers.append(GraphConvolution(inDim, hidDim, bias=bias, dropout=dropout))
		for _ in range(numOfGCNLayers - 2):
			self.GCLayers.append(GraphConvolution(hidDim, hidDim, bias=bias, dropout=dropout))

		self.GCLayers.append(GraphConvolution(hidDim, outDim, bias=bias, dropout=0.0))

		# self.post_mp = nn.Sequential(
		# 	nn.Linear(hidDim, hidDim), nn.Dropout(dropout),
		# 	nn.Linear(hidDim, outDim))

	def forward(self, input, adj):
		tensor = input
		for GC in self.GCLayers:
			tensor = GC(tensor, adj)

		# output = self.post_mp(tensor)
		output = tensor

		return F.log_softmax(output, dim=1)
