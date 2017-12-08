import mnist
import numpy as np
from scipy.misc import imsave
import os, sys


def save_data(labels, images, output_dir):
	output_dirs = [
		os.path.join(output_dir, str(i)) for i in xrange(10)
	]

	for output_dir in output_dirs:
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)


	for (i, label) in enumerate(labels):
		output_filename = os.path.join(output_dirs[label], str(i) + '.png')
		print "Writing to ", output_filename

		img = images[i].squeeze()
		imsave(output_filename, img)


output_dir = sys.argv[1]
val_split = float(sys.argv[2])

print "Downloading training data"
train_labels, train_images = mnist.train_labels(), mnist.train_images()

trainset_len = train_labels.shape[0]
val_count = int(val_split * trainset_len)

idxs = np.random.permutation(np.arange(trainset_len))
val_idxs = idxs[:val_count]
train_idxs = idxs[val_count:]

val_labels, val_images = np.take(train_labels, val_idxs), np.take(train_images, val_idxs, axis=0)
train_labels, train_images = np.take(train_labels, train_idxs), np.take(train_images, train_idxs, axis=0)

print "Downloading test data"
test_labels, test_images = mnist.test_labels(), mnist.test_images()


val_output = os.path.join(output_dir, 'val')
train_output = os.path.join(output_dir, 'train')
test_output = os.path.join(output_dir, 'test')

print "Saving Validation data"
save_data(val_labels, val_images, val_output)

print "Saving Training data"
save_data(train_labels, train_images, train_output)

print "Saving Test data"
save_data(test_labels, test_images, test_output)