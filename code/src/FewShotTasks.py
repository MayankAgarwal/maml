import os
import random
import numpy as np
import torch

class OmniglotTask(object):

	def __init__(self, root, num_cls, num_inst, split='train'):

		self.dataset = 'omniglot'
		self.root = '{}/images_background'.format(root) if split=='train' else '{}/images_evaluation'.format(root)
		self.num_cls = num_cls
		self.num_inst = num_inst

		languages = os.listdir(self.root)
		chars = []

		for l in languages:
			chars += [os.path.join(l, x) for x in os.listdir(os.path.join(self.root, l))]

		random.shuffle(chars)
		classes = chars[:num_cls]
		labels = np.array(range(len(classes)))
		labels = dict(zip(classes, labels))

		instances = dict()

		self.train_ids = []
		self.val_ids = []

		for c in classes:
			temp = [os.path.join(c, x) for x in os.listdir(os.path.join(self.root, c))]
			instances[c] = random.sample(temp, len(temp))

			self.train_ids += instances[c][:num_inst]
			self.val_ids += instances[c][num_inst: num_inst*2]

		self.train_labels = [labels[self.get_class(x)] for x in self.train_ids]
		self.val_labels = [labels[self.get_class(x)] for x in self.val_ids]

	def get_class(self, instance):
		return os.path.join(*instance.split('/')[:-1])


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

		self.label_map = dict(zip(range(10), np.random.permutation(np.array(range(10)))))

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

class SinusoidTask(object):
	""" Sinusoidal regression task """

	def __init__(self, num_inst, split='train'):
		
		self.dataset = 'sinusoid_reg'
		self.root = None
		self.split = split
		self.num_inst = num_inst
		self.num_cls = 1

		self.train_ids, self.train_labels = self.__get_samples()
		self.val_ids, self.val_labels = self.__get_samples()

	def __get_samples(self):

		def compute_sin(amp, phase, x):
			return amp * np.sin(x + phase)

		amp = np.random.uniform(0.1, 5.0)
		phase = np.random.uniform(0, np.pi)

		x = np.random.uniform(-5.0, 5.0, size=self.num_inst)
		y = compute_sin(amp, phase, x)

		return x, y