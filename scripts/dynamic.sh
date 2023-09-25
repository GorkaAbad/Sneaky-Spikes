# N-MNIST
python dynamic.py --dataset mnist --cupy --epochs 10 --train_epochs 1 --alpha 0.5 --beta 0.01

# CIFAR-10
python dynamic.py --dataset cifar10 --cupy --epochs 28 --train_epochs 1 --alpha 0.5 --beta 0.01

# Gesture
python dynamic.py --dataset gesture --cupy --epochs 64 --train_epochs 1 --alpha 0.5 --beta 0.01

# Caltech-101
python dynamic.py --dataset caltech --cupy --epochs 30 --train_epochs 1 --alpha 0.5 --beta 0.01