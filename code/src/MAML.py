import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import SGD, Adam
from torch.nn.modules.loss import CrossEntropyLoss

from FC_Net import FC_Net
from FC_Metanet import FC_MetaNet
from score_utils import *
from dataloader_utils import *
from FewShotTasks import MNISTTask
from FewShotDatasets import MNIST

import sys, os

class MAML(object):

	def __init__(self,
				dataset, net_type, num_classes, num_insts, 
				meta_batch_size, meta_step_size,
				inner_batch_size, inner_step_size, 
				num_updates, num_inner_updates, loss_fn):

		self.dataset = dataset
		self.net_type = net_type
		self.num_classes = num_classes
		self.num_insts = num_insts
		self.meta_batch_size = meta_batch_size
		self.meta_step_size =  meta_step_size
		self.inner_batch_size = inner_batch_size
		self.inner_step_size = inner_step_size
		self.num_updates = num_updates
		self.num_inner_updates = num_inner_updates
		self.loss_fn = loss_fn
		self.is_cuda = torch.cuda.is_available()
		self.dtype = torch.cuda.FloatTensor if self.is_cuda else torch.FloatTensor
		self.longTensor = torch.cuda.LongTensor if self.is_cuda else torch.LongTensor

		self.__init_nets__()
		self.opt = Adam(self.net.parameters(), lr=meta_step_size)
		
	def __init_nets__(self):

		if self.net_type == 'fc-full':
			self.__init_fc_nets__()

	def __get_test_net__(self):

		if self.net_type == 'fc-full':
			return self.__get_fc_test_net__()


	def __init_fc_nets__(self):

		if self.dataset == 'mnist':
			input_dim = 28*28


		self.net = FC_Net(self.num_classes, input_dim, self.loss_fn, self.dtype)
		self.metanet = FC_MetaNet(self.num_classes, input_dim, self.loss_fn, self.num_inner_updates, self.inner_step_size, self.inner_batch_size, self.meta_batch_size, self.dtype)

		if self.is_cuda:
			self.net.cuda()
			self.metanet.cuda()

	def __get_fc_test_net__(self):

		if self.dataset == "mnist":
			input_dim = 28*28

		test_net = FC_Net(self.num_classes, input_dim, self.loss_fn, self.dtype)
		return test_net


	def get_task(self, root, n_cls, n_inst, split='train'):

		if 'mnist' in root:
			return MNISTTask(root, n_cls, n_inst, split)


	def test(self):

		test_net = self.__get_test_net__()
		mtr_loss, mtr_acc, mval_loss, mval_acc = 0.0, 0.0, 0.0, 0.0

		for _ in xrange(10):
			test_net.load_state_dict(self.net.state_dict())
			if self.is_cuda:
				test_net.cuda()

			test_opt = SGD(test_net.parameters(), lr=self.inner_step_size)
			task = self.get_task('../data/{}'.format(self.dataset), self.num_classes, self.num_insts, split='test')
			train_loader = get_data_loader(task, self.inner_batch_size, split='train')

			for i in xrange(self.num_inner_updates):
				in_, target = train_loader.__iter__().next()
				loss, _ = forward_pass(test_net, in_, target, dtype=self.dtype, long_dtype=self.longTensor)
				test_opt.zero_grad()
				loss.backward()
				test_opt.step()

			tloss, tacc = evaluate(test_net, train_loader)
			val_loader = get_data_loader(task, self.inner_batch_size, split='val')
			vloss, vacc = evaluate(test_net, val_loader)

			mtr_loss += tloss
			mtr_acc += tacc
			mval_acc += vacc
			mval_loss += vloss

		mtr_loss /= 10.0
		mtr_acc /= 10.0
		mval_acc /= 10.0
		mval_loss /= 10.0

		print '-' * 100
		print '\t Meta train: ', mtr_loss, mtr_acc
		print '\t Meta val: ', mval_loss, mval_acc
		print '-' * 100

		del test_net
		return mtr_loss, mtr_acc, mval_loss, mval_acc

	def train(self, expid):

		tr_loss, tr_acc, val_loss, val_acc = [], [], [], []
		mtr_loss, mtr_acc, mval_loss, mval_acc = [], [], [], []

		for it in xrange(self.num_updates):

			mt_loss, mt_acc, mv_loss, mv_acc = self.test()
			mtr_loss.append(mt_loss)
			mtr_acc.append(mt_acc)
			mval_loss.append(mv_loss)
			mval_acc.append(mv_acc)

			grads = []
			tloss, tacc, vloss, vacc = 0.0, 0.0, 0.0, 0.0

			for i in xrange(self.meta_batch_size):
				task = self.get_task('../data/{}'.format(self.dataset), self.num_classes, self.num_insts)
				self.metanet.load_state_dict(self.net.state_dict())
				metrics, g = self.metanet.forward(task)
				(trl, tra, vall, vala) = metrics
				grads.append(g)

				tloss += trl
				tacc += tra
				vloss += vall
				vacc += vala

			print "Meta update: ", it
			self.meta_update(task, grads)

			if it%500 == 0:
				torch.save(self.net.state_dict(), '../output/{}/train_iter_{}.pth'.format(expid, it))

			tr_loss.append(tloss/self.meta_batch_size)
			tr_acc.append(tacc/self.meta_batch_size)
			val_loss.append(vall/self.meta_batch_size)
			val_acc.append(vala/self.meta_batch_size)

			np.save('../output/{}/tr_loss.npy'.format(expid), np.array(tr_loss))
			np.save('../output/{}/tr_acc.npy'.format(expid), np.array(tr_acc))
			np.save('../output/{}/val_loss.npy'.format(expid), np.array(val_loss))
			np.save('../output/{}/val_acc.npy'.format(expid), np.array(val_acc))

			np.save('../output/{}/meta_tr_loss.npy'.format(expid), np.array(mtr_loss))
			np.save('../output/{}/meta_tr_acc.npy'.format(expid), np.array(mtr_acc))
			np.save('../output/{}/meta_val_loss.npy'.format(expid), np.array(mval_loss))
			np.save('../output/{}/meta_val_acc.npy'.format(expid), np.array(mval_acc))


	def meta_update(self, task, metagrads):

		print '\nMeta-update \n'

		loader = get_data_loader(task, self.inner_batch_size, split='val')
		in_, target = loader.__iter__().next()

		loss, out = forward_pass(self.net, in_, target)

		gradients = {k: sum(d[k] for d in metagrads) for k in metagrads[0].keys()}

		hooks = []
		for (k, v) in self.net.named_parameters():
			def get_closure():
				key = k
				def replace_grad(grad):
					return gradients[key]
				return replace_grad

			hooks.append(v.register_hook(get_closure()))

		self.opt.zero_grad()
		loss.backward()
		self.opt.step()

		for h in hooks:
			h.remove()


def main(expid, dataset, net_type, num_cls, num_insts, batch, m_batch, num_updates, num_inner_updates, lr, meta_lr):

	output = '../output/{}'.format(expid)

	try:
		os.makedirs(output)
	except Exception as err:
		print err

	loss_fn = CrossEntropyLoss()
	learner = MAML(dataset, net_type, num_cls, num_insts, m_batch, float(meta_lr), batch, float(lr), num_updates, num_inner_updates, loss_fn)
	learner.train(expid)

if __name__ == "__main__":
	expid = sys.argv[1]
	dataset = sys.argv[2]	# mnist
	net_type = sys.argv[3]	# fc-full
	num_cls = int(sys.argv[4])
	num_insts = int(sys.argv[5])
	batch = int(sys.argv[6])	# Inner batch size
	m_batch = int(sys.argv[7])	# Meta batch size
	num_updates = int(sys.argv[8]) # Number of updates
	num_inner_updates = int(sys.argv[9])	# Number of inner updates
	lr = float(sys.argv[10])	# Learning rate
	meta_lr = float(sys.argv[11])	# Meta learning rate

	main(expid, dataset, net_type, num_cls, num_insts, batch, m_batch, num_updates, num_inner_updates, lr, meta_lr)

