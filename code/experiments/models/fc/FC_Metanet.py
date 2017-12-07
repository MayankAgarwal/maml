import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable

from FC_Net import FC_Net

class FC_MetaNet(FC_Net):
	"""
	Meta-network FC Net
	"""

	def __init__(self, input_dim, num_classes, loss_fn, num_updates, step_size, batch_size, meta_batch_size, dtype=torch.FloatTensor):

		super(FC_MetaNet, self).__init__(num_classes, input_dim, loss_fn, dtype)

		self.num_updates = num_updates
		self.step_size = step_size
		self.batch_size = batch_size
		self.meta_batch_size = meta_batch_size
		self.dtype = dtype

	def net_forward(self, x, weights=None):
		return super(FC_MetaNet, self).forward(x, weights)

	def forward_pass(self, in_, target, weights=None):
		input_var = Variable(in_).type(self.dtype)
		target_var = Variable(target).type(self.dtype)
		out = self.net_forward(input_var, weights)
		loss = self.loss_fn(out, target_var)
		return loss, out
