import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

class FewShotDataset(data.Dataset):

	def __init__(self, task, split='train', transforms=None, target_transforms=None):

		self.task = task
		self.split = split
		self.transforms = transforms
		self.target_transforms = target_transforms

		self.root = self.task.root
		self.input_ids = self.task.train_ids if self.split=='train' else self.task.val_ids
		self.labels = self.task.train_labels if self.split=='train' else self.task.val_labels

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		raise NotImplementedError("This is abstract class")


# TODO : Add Omniglot dataset 


class MNIST(FewShotDataset):

	def __init__(self, *args, **kwargs):
		super(MNIST, self).__init__(*args, **kwargs)

	def load_image(self, idx):
		im = Image.open('{}/{}.png'.format(self.root, idx))#.convert('RGB')
		im = np.array(im, dtype=np.float32)
		return im

	def __getitem__(self, idx):
		img_id = self.input_ids[idx]
		img = self.load_image(img_id)

		if self.transforms is not None:
			img = self.transforms(img)

		target = self.labels[idx]
		if self.target_transforms is not None:
			target = self.target_transforms(target)

		return img, target