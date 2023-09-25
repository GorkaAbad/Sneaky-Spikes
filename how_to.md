# How to run the code

## Requirements

Install SpikingJelly repo from [here](https://spikingjelly.readthedocs.io).

Install the requirements:
```bash
pip install -r requirements.txt
```

## Hardware requirements

In order to run the code a GPU is strongly recommended. 
The code is tested on a machine with 1 NVIDIA A100 GPUs with 40GB.

## Examples

Script examples for different datasets are provided in the `scripts` folder. 

Get help:
```bash
python main.py --help

usage: main.py [-h] [--model MODEL] [--dataset DATASET] [--lr LR] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--T T]
               [--amp] [--cupy] [--loss {mse,cross}] [--optim {adam,sgd}] [--trigger_label TRIGGER_LABEL]
               [--polarity {0,1,2,3}] [--trigger_size TRIGGER_SIZE] [--epsilon EPSILON]
               [--pos {top-left,top-right,bottom-left,bottom-right,middle,random}] [--type {static,moving,smart}]
               [--n_masks N_MASKS] [--least] [--most_polarity] [--momentum MOMENTUM] [--data_dir DATA_DIR]
               [--save_path SAVE_PATH] [--model_path MODEL_PATH] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model name to use
  --dataset DATASET     Dataset to use
  --lr LR               Learning rate
  --batch_size BATCH_SIZE
                        Batch size
  --epochs EPOCHS       Number of epochs
  --T T                 simulating time-steps
  --amp                 Use automatic mixed precision training
  --cupy                Use cupy
  --loss {mse,cross}    Loss function
  --optim {adam,sgd}    Optimizer
  --trigger_label TRIGGER_LABEL
                        The index of the trigger label
  --polarity {0,1,2,3}  The polarity of the trigger
  --trigger_size TRIGGER_SIZE
                        The size of the trigger as the percentage of the image size
  --epsilon EPSILON     The percentage of poisoned data
  --pos {top-left,top-right,bottom-left,bottom-right,middle,random}
                        The position of the trigger
  --type {static,moving,smart}
                        The type of the trigger
  --n_masks N_MASKS     The number of masks. Only if the trigger type is smart
  --least               Use least active area for smart attack
  --most_polarity       Use most active polarity in the area for smart attack
  --momentum MOMENTUM   Momentum
  --data_dir DATA_DIR   Data directory
  --save_path SAVE_PATH
                        Path to save the experiments
  --model_path MODEL_PATH
                        Clean model path for dynamic attack
  --seed SEED           Random seed

```

## Static Triggers

Example of running a static trigger attack on N-MNIST dataset, in the top-left corner, with 10% of the data poisoned, polarity 1, with the trigger size of 10% of the image size:

```bash	
python main.py --dataset mnist --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --type static --cupy --epochs 10
```

## Moving Triggers

Example of running a moving trigger attack on N-MNIST dataset, in the top-left corner, with 10% of the data poisoned, polarity 1, with the trigger size of 10% of the image size:

```bash
python main.py --dataset mnist --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --type moving --cupy --epochs 10
```

## Smart Triggers

Example of running a smart trigger attack on N-MNIST dataset, in the the least important area, with 10% of the data poisoned, with the trigger size of 10% of the image size:

```bash
python main.py --dataset mnist --trigger_size 0.1 --epsilon 0.1 --type smart --least --cupy --epochs 10 
```

## Dynamic Triggers

Get help:
```bash
usage: dynamic.py [-h] [--dataset DATASET] [--lr LR] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                  [--train_epochs TRAIN_EPOCHS] [--T T] [--amp] [--cupy] [--loss {mse,cross}] [--optim {adam,sgd}]
                  [--trigger_label TRIGGER_LABEL] [--momentum MOMENTUM] [--alpha ALPHA] [--beta BETA]
                  [--data_dir DATA_DIR] [--save_path SAVE_PATH] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset to use
  --lr LR               Learning rate
  --batch_size BATCH_SIZE
                        Batch size
  --epochs EPOCHS       Number of epochs
  --train_epochs TRAIN_EPOCHS
                        Number of epochs
  --T T                 simulating time-steps
  --amp                 Use automatic mixed precision training
  --cupy                Use cupy
  --loss {mse,cross}    Loss function
  --optim {adam,sgd}    Optimizer
  --trigger_label TRIGGER_LABEL
                        The index of the trigger label
  --momentum MOMENTUM   Momentum
  --alpha ALPHA         Alpha
  --beta BETA           Beta. Gamma in the paper
  --data_dir DATA_DIR   Data directory
  --save_path SAVE_PATH
                        Path to save the experiments
  --seed SEED           Random seed
```

Example of running a dynamic trigger attack on N-MNIST dataset:

```bash
python dynamic.py --dataset mnist --cupy --epochs 10 --train_epochs 1 --alpha 0.5 --beta 0.01
```