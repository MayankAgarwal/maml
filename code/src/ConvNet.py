import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
import math

class ConvNet(nn.Module):
	"""
	Mini Conv master net for MAML
	"""

	def __init__(self, num_classes, num_in_channels, loss_fn, dtype=torch.FloatTensor):
		
		super(ConvNet, self).__init__()
		
		self.features = nn.Sequential(OrderedDict([
				('conv1', nn.Conv2d(num_in_channels, 64, 3)),
				('bn1', nn.BatchNorm2d(64, momentum=1, affine=True)),
				('relu1', nn.ReLU(inplace=True)),
				('pool1', nn.MaxPool2d(2,2)),
				('conv2', nn.Conv2d(64,64,3)),
				('bn2', nn.BatchNorm2d(64, momentum=1, affine=True)),
				('relu2', nn.ReLU(inplace=True)),
				('pool2', nn.MaxPool2d(2,2)),
				('conv3', nn.Conv2d(64,64,3)),
				('bn3', nn.BatchNorm2d(64, momentum=1, affine=True)),
				('relu3', nn.ReLU(inplace=True)),
				('pool3', nn.MaxPool2d(2,2))
		]))
		self.add_module('fc', nn.Linear(64, num_classes))
		
		# Define loss function
		self.loss_fn = loss_fn

		self.dtype = dtype

		# Initialize weights
		self._init_weights()

	def forward(self, x, weights=None):
		''' Define what happens to data in the net '''
		if weights == None:
			x = self.features(x)
			x = x.view(x.size(0), 64)
			x = self.fc(x)
		else:
			x = self.__conv2d(x, weights['features.conv1.weight'], weights['features.conv1.bias'])
			x = self.__batchnorm(x, weight = weights['features.bn1.weight'], bias = weights['features.bn1.bias'], momentum=1)
			x = self.__relu(x)
			x = self.__maxpool(x, kernel_size=2, stride=2) 
			x = self.__conv2d(x, weights['features.conv2.weight'], weights['features.conv2.bias'])
			x = self.__batchnorm(x, weight = weights['features.bn2.weight'], bias = weights['features.bn2.bias'], momentum=1)
			x = self.__relu(x)
			x = self.__maxpool(x, kernel_size=2, stride=2) 
			x = self.__conv2d(x, weights['features.conv3.weight'], weights['features.conv3.bias'])
			x = self.__batchnorm(x, weight = weights['features.bn3.weight'], bias = weights['features.bn3.bias'], momentum=1)
			x = self.__relu(x)
			x = self.__maxpool(x, kernel_size=2, stride=2) 
			x = x.view(x.size(0), 64)
			x = self.__linear(x, weights['fc.weight'], weights['fc.bias'])
		return x

	def net_forward(self, x, weights=None):
		return self.forward(x, weights)
	
	def _init_weights(self):
		''' Set weights to Gaussian, biases to zero '''
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data = torch.ones(m.bias.data.size())

	def __linear(self, input, weight, bias=None):
			return F.linear(input, weight.type(self.dtype), bias.type(self.dtype))

	def __batchnorm(self, input, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-5, momentum=0.1):
		running_mean = torch.zeros(np.prod(np.array(input.data.size()[1]))).type(self.dtype)
		running_var = torch.ones(np.prod(np.array(input.data.size()[1]))).type(self.dtype)
		return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

	def __relu(self, input):
		return F.threshold(input, 0, 0, inplace=True)

	def __conv2d(self, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
		return F.conv2d(input, weight.type(self.dtype), bias.type(self.dtype), stride, padding, dilation, groups)

	def __maxpool(self, input, kernel_size, stride=None):
		return F.max_pool2d(input, kernel_size, stride)
