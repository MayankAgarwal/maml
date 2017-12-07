import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

class FewShotDataset(data.Dataset):

	def __init__(self, task, split='train', transform=None, target_transform=None):

		self.task = task
		self.split = split
		self.transform = transform
		self.target_transform = target_transform

		self.root = self.task.root
		self.input_ids = self.task.train_ids if self.split=='train' else self.task.val_ids
		self.labels = self.task.train_labels if self.split=='train' else self.task.val_labels

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		raise NotImplementedError("This is abstract class")


# TODO : Add Omniglot dataset 


class MNIST(data.Dataset):

	def __init__(self, *args, **kwargs):
		super(MNIST, self).__init__(*args, **kwargs)

	def load_image(self, idx):
		im = Image.open('{}/{}.png'.format(self.root, idx)).convert('RGB')
		im = np.array(im, dtype=np.float32)
		return im

	def __getitem__(self, idx):
		img_id = self.input_ids[idx]
		img = self.load_image(img_id)

		if self.transform is not None:
			img = self.transform(img)

		target = self.labels[idx]
		if self.target_transform is not None:
			target = self.target_transform(target):

		return img, target