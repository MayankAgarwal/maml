import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class MNIST(object):
    
    @staticmethod
    def load_data(batch_size, valid_size=0.2):
        
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        
        train_data = torchvision.datasets.MNIST('data/mnist', train=True, transform=transform, download=True)
        valid_data = torchvision.datasets.MNIST('data/mnist', train=True, transform=transform)
        test_data = torchvision.datasets.MNIST('data/mnist', train=False, transform=transform)
        
        num_train = len(train_data)
        idxs = range(num_train)
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(idxs)
        
        train_idxs, valid_idxs = idxs[split:], idxs[:split]
        
        train_subsampler = SubsetRandomSampler(train_idxs)
        valid_subsampler = SubsetRandomSampler(valid_idxs)
        
        train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_subsampler, num_workers=4)
        valid_dl = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, sampler=valid_subsampler, num_workers=4)
        test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
        
        return train_dl, valid_dl, test_dl