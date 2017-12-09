import numpy as np

import torch
from torch.autograd import Variable

def count_correct(pred, target):
	pairs = [int(x==y) for (x, y) in zip(pred, target)]
	return sum(pairs)

def forward_pass(net, in_, target, weights=None, dtype=torch.FloatTensor, long_dtype=torch.LongTensor):

	input_var = Variable(in_).type(dtype)
	target_var = Variable(target).type(long_dtype)
	out = net.net_forward(input_var, weights)
	loss = net.loss_fn(out, target_var)
	return loss, out

def evaluate(net, loader, weights=None):
	
	num_correct, loss = 0, 0

	for i, (in_, target) in enumerate(loader):
		batch_size = in_.numpy().shape[0]
		l, out = forward_pass(net, in_, target, weights)
		loss += l.data.cpu().numpy()[0]
		num_correct += count_correct(np.argmax(out.data.cpu().numpy(), axis=1), target.numpy())
	return float(loss)/len(loader), float(num_correct)/(len(loader)*batch_size)