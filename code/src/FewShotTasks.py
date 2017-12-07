import os
import random
import numpy as np
import torch

# TODO : Omniglot task
#  

class MNISTTask(object):
	""" Create a few-shot task using MNIST dataset """

	def __init__(self, root, num_cls, num_inst, split='train'):

		self.dataset = 'mnist'
		self.root = root + '/' + split
		self.split = split
		self.num_inst = num_inst
		self.num_cls = num_cls
		all_ids = []

		for i in xrange(10):
			d = os.path.join(root, self.split, str(i))
			files = os.listdir(d)
			all_ids.append([str(i) + '/' + f[:-4] for f in files])

		self.label_map = dict(zip(range(10, np.random.permutation(np.array(range(10))))))

		self.train_ids = []
		self.val_ids = []

		for i in xrange(10):
			permutation = list(np.random.permutation(np.array(range(len(all_ids[i])))))[:num_inst*2]

			self.train_ids.extend([all_ids[i][j] for j in permutation[:num_inst]])
			self.train_labels = self.relabel(self.train_ids)
			self.val_ids.extend([all_ids[i][j] for j in permutation[num_inst:]])
			self.val_labels = self.relabel(self.val_ids)


	def relabel(self, img_ids):

		orig_labels = [int(x[0]) for x in img_ids]
		return [self.label_map[x] for x in orig_labels]