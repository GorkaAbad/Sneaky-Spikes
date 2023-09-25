# N-MNIST
python main.py --dataset mnist --n_masks 2 --trigger_size 0.1 --epsilon 0.1 --type smart --cupy --epochs 10

# CIFAR-10
python main.py --dataset cifar10 --n_masks 2 --trigger_size 0.1 --epsilon 0.1 --type smart --cupy --epochs 28

# Gesture
python main.py --dataset gesture --n_masks 2 --trigger_size 0.1 --epsilon 0.1 --type smart --cupy --epochs 64

# Caltech-101
python main.py --dataset caltech --n_masks 2 --trigger_size 0.1 --epsilon 0.1 --type smart --cupy --epochs 30