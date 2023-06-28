from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.es_imagenet import ESImageNet
from spikingjelly.datasets import split_to_train_test_set
import os


def get_dataset(dataset, frames_number, data_dir):

    data_dir = os.path.join(data_dir, dataset)

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

        # cifar = os.path.join(data_dir, 'cifar10.pt')
        # if not os.path.exists(cifar):
        #     # TODO: Since this is slow, consider saving the dataset
        #     train_set, test_set = split_to_train_test_set(
        #         origin_dataset=dataset, train_ratio=0.9, num_classes=10)
        #     torch.save({'train': train_set, 'test': test_set}, cifar)

        # else:
        #     data = torch.load(cifar)
        #     train_set = data['train']
        #     test_set = data['test']

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

    else:
        raise ValueError(f'{dataset} is not supported')

    # train_set.samples = train_set.samples[:100]
    # test_set.samples = test_set.samples[:5000]
    # train_set.targets = train_set.targets[:100]
    # test_set.targets = test_set.targets[:5000]
    return train_set, test_set
