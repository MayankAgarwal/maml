import torch
import torch.nn as nn
from torch.nn import Functional as F
from collections import OrderedDict
import numpy as np

class FC_Net(nn.Module):
	"""
	Fully-Connected master net for MAML
	"""

	def __init__(self, num_classes, input_dim, loss_fn, dtype=torch.FloatTensor):

		super(FC_Net, self).__init__()

		self.num_classes = num_classes
		self.input_dim = input_dim

		self.layers = nn.Sequential(
			OrderedDict([
				('fc1', nn.Linear(input_dim, 256)),
				('bn1', nn.BatchNorm1d(256, affine=True)),
				('relu1', nn.ReLU(inplace=True)),

				('fc2', nn.Linear(256, 128)),
				('bn2', nn.BatchNorm1d(128, affine=True)),
				('relu2', nn.ReLU(inplace=True)),

				('fc3', nn.Linear(128, 64)),
				('bn3', nn.BatchNorm1d(64, affine=True)),
				('relu3', nn.ReLU(inplace=True)),

				('fc4', nn.Linear(64, 64)),
				('bn4', nn.BatchNorm1d(64, affine=True)),
				('relu4', nn.ReLU(inplace=True))
				])
			)

		self.add_module('fc_out', nn.Linear(64, num_classes))

		self.loss_fn = loss_fn
		self.dtype = dtype

	def forward(self, x, weights=None):

		x = x.view(-1, self.input_dim)

		if weights == None:
			# Main net trains here
			x = self.layers(x)
			x = self.fc(x)
		else:
			# This code block is used by the meta-network
			x = self.__linear(x, weight=weights['layers.fc1.weight'], bias=weights['layers.fc1.bias'])
			x = self.__batchnorm(x, weight=weights['layers.bn1.weight'], bias=weights['layers.bn1.bias'], momentum=1)
			x = self.__relu(x)

			x = self.__linear(x, weight=weights['layers.fc2.weight'], bias=weights['layers.fc2.bias'])
			x = self.__batchnorm(x, weight=weights['layers.bn2.weight'], bias=weights['layers.bn2.bias'], momentum=1)
			x = self.__relu(x)

			x = self.__linear(x, weight=weights['layers.fc3.weight'], bias=weights['layers.fc3.bias'])
			x = self.__batchnorm(x, weight=weights['layers.bn3.weight'], bias=weights['layers.bn3.bias'], momentum=1)
			x = self.__relu(x)

			x = self.__linear(x, weight=weights['layers.fc4.weight'], bias=weights['layers.fc4.bias'])
			x = self.__batchnorm(x, weight=weights['layers.bn4.weight'], bias=weights['layers.bn4.bias'], momentum=1)
			x = self.__relu(x)

			x = self.__linear(x, weights['fc_out.weight'], weights['fc_out.bias'])

		return x

	def net_forward(self, x, weights=None):
		return self.forward(x, weights)

	def __linear(self, input, weight, bias=None):
			return F.linear(input, weight.type(self.dtype), bias.type(self.dtype))

	def __batchnorm(self, input, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-5, momentum=0.1):
		running_mean = torch.zeros(np.prod(np.array(input.data.size()[1]))).type(self.dtype)
		running_var = torch.ones(np.prod(np.array(input.data.size()[1]))).type(self.dtype)
		return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

	def __relu(self, input):
		return F.threshold(input, 0, 0, inplace=True)

