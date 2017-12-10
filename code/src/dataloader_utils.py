import numpy as np
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms

from FewShotDatasets import MNIST, Sinusoidal, Omniglot

class ClassBalancedSampler(Sampler):

	def __init__(self, num_cls, num_inst, batch_cutoff=None):

		self.num_cls = num_cls
		self.num_inst = num_inst
		self.batch_cutoff = batch_cutoff

	def __iter__(self):

		batches = [[i + j*self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cls)]
		batches = [[batches[j][i] for j in range(self.num_cls)] for i in range(self.num_inst)]

		# Shuffle so that the class order is changed
		for sublist in batches:
			random.shuffle(sublist)

		if self.batch_cutoff is not None:
			random.shuffle(batches)
			batches = batches[:self.batch_cutoff]

		batches = [item for sublist in batches for item in sublist]

		return iter(batches)

	def __len__(self):
		return 1


def get_data_loader(task, batch_size=1, split='train'):
	# NOTE: Batch size is num instances per class
	
	if task.dataset == 'mnist':
		#normalize = transforms.Normalize(mean=[0.13066, 0.13066, 0.13066], std=[0.30131, 0.30131, 0.30131])
		dset = MNIST(task, transforms=None, split=split)
	elif task.dataset == 'sinusoid_reg':
		dset = Sinusoidal(task, transforms=None, split=split)
	elif task.dataset == 'omniglot':
		resize = transforms.Resize((28, 28)) # Images are
		dset = Omniglot(task, transforms=transforms.Compose([resize, transforms.ToTensor()]), split=split)

	batch_cutoff = None if split!='train' else batch_size

	sampler = ClassBalancedSampler(task.num_cls, task.num_inst, batch_cutoff=batch_cutoff)
	loader = DataLoader(dset, batch_size=batch_size*task.num_cls, sampler=sampler, num_workers=1)
	return loader

