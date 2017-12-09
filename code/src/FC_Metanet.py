import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable

from score_utils import *
from dataloader_utils import *

from FC_Net import FC_Net

class FC_MetaNet(FC_Net):
	"""
	Meta-network FC Net
	"""

	def __init__(self, num_classes, input_dim, loss_fn, num_meta_updates, step_size, batch_size, meta_batch_size, dtype=torch.FloatTensor, long_dtype=torch.LongTensor):

		super(FC_MetaNet, self).__init__(num_classes, input_dim, loss_fn, dtype)

		self.num_meta_updates = num_meta_updates
		self.step_size = step_size
		self.batch_size = batch_size
		self.meta_batch_size = meta_batch_size
		self.dtype = dtype
		self.long_dtype = long_dtype

	def net_forward(self, x, weights=None):
		return super(FC_MetaNet, self).forward(x, weights)

	def forward_pass(self, in_, target, weights=None):
		input_var = Variable(in_).type(self.dtype)
		target_var = Variable(target).type(self.long_dtype)
		out = self.net_forward(input_var, weights)
		loss = self.loss_fn(out, target_var)
		return loss, out

	def forward(self, task):

		train_loader = get_data_loader(task, self.batch_size)
		val_loader = get_data_loader(task, self.batch_size, split='val')

		tr_pre_loss, tr_pre_acc = evaluate(self, train_loader)
		val_pre_loss, val_pre_acc = evaluate(self, val_loader)
		
		metanet_weights = OrderedDict((name, param) for (name, param) in self.named_parameters())

		for i in xrange(self.num_meta_updates):
			print "Meta-net update step: %d" % i

			in_, target = train_loader.__iter__().next()

			if i==0:
				loss, _ = self.forward_pass(in_, target)
				grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
			else:
				loss, _ = self.forward_pass(in_, target, metanet_weights)
				grads = torch.autograd.grad(loss, metanet_weights.values(), create_graph=True)

			metanet_weights = OrderedDict((name, param - self.step_size*grad) for ((name, param), grad) in zip(metanet_weights.items(), grads))

		tr_post_loss, tr_post_acc = evaluate(self, train_loader, metanet_weights)
		val_post_loss, val_post_acc = evaluate(self, val_loader, metanet_weights)

		print "Meta-net step train loss. Pre: ", tr_pre_loss, ", Post: ", tr_post_loss
		print "Meta-net step train acc. Pre: ", tr_pre_acc, ", Post: ", tr_post_acc
		print "Meta-net step val loss. Pre: ", val_pre_loss, ", Post: ", val_post_loss
		print "Meta-net step val acc. Pre: ", val_pre_acc, ", Post: ", val_post_acc

		# Compute the meta gradient
		in_, target = val_loader.__iter__().next()
		loss, _ = self.forward_pass(in_, target, metanet_weights)
		loss = loss / self.meta_batch_size
		grads = torch.autograd.grad(loss, self.parameters())
		meta_grads = {name:g for ((name, _), g) in zip(self.named_parameters(), grads)}
		metrics = (tr_post_loss, tr_post_acc, val_post_loss, val_post_acc)
		return metrics, meta_grads
