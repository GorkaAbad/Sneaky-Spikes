from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.es_imagenet import ESImageNet
from spikingjelly.datasets import split_to_train_test_set
from spikingjelly.datasets.n_caltech101 import NCaltech101
import os
import torch


def get_dataset(dataset, frames_number, data_dir):

    data_dir = os.path.join(data_dir, dataset)

    path_train = os.path.join(data_dir, f'{frames_number}_train_split.pt')
    path_test = os.path.join(data_dir, f'{frames_number}_test_split.pt')

    if dataset == 'gesture':
        transform = None

        train_set = DVS128Gesture(
            data_dir, train=True, data_type='frame', split_by='number', frames_number=frames_number, transform=transform)
        test_set = DVS128Gesture(data_dir, train=False,
                                 data_type='frame', split_by='number', frames_number=frames_number, transform=transform)

    elif dataset == 'cifar10':

        # Split by number as in: https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron

        dataset = CIFAR10DVS(data_dir, data_type='frame',
                             split_by='number', frames_number=frames_number)

        train_set, test_set = split_to_train_test_set(
            origin_dataset=dataset, train_ratio=0.9, num_classes=10)

    elif dataset == 'mnist':

        train_set = NMNIST(data_dir, train=True, data_type='frame',
                           split_by='number', frames_number=frames_number)

        test_set = NMNIST(data_dir, train=False, data_type='frame',
                          split_by='number', frames_number=frames_number)

    elif dataset == 'imagenet':
        train_set = ESImageNet(data_dir, train=True, data_type='frame',
                               split_by='number', frames_number=frames_number)
        test_set = ESImageNet(data_dir, train=False, data_type='frame',
                              split_by='number', frames_number=frames_number)
    elif dataset == 'caltech':

        dataset = NCaltech101(data_dir, data_type='frame',
                            split_by='number', frames_number=frames_number)

        if os.path.exists(path_train) and os.path.exists(path_test):
            train_set = torch.load(path_train)
            test_set = torch.load(path_test)
        else:
            train_set, test_set = split_to_train_test_set(
                origin_dataset=dataset, train_ratio=0.9, num_classes=101)

            torch.save(train_set, path_train)
            torch.save(test_set, path_test)
                
    else:
        raise ValueError(f'{dataset} is not supported')

    return train_set, test_set
