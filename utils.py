from traceback import print_tb
from matplotlib.pyplot import sca
import torch.nn as nn
from torch import optim
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import seaborn as sns
import csv
from spikingjelly.activation_based import functional
from torch.cuda import amp
import torch.nn.functional as F


def loss_picker(loss):
    '''
    Select the loss function
    Parameters:
        loss (str): name of the loss function
    Returns:
        loss_function (torch.nn.Module): loss function
    '''
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("Automatically assign mse loss function to you...")
        criterion = nn.MSELoss()

    return criterion


def optimizer_picker(optimization, param, lr, momentum, epochs):
    '''
    Select the optimizer
    Parameters:
        optimization (str): name of the optimization method
        param (list): model's parameters to optimize
        lr (float): learning rate
    Returns:
        optimizer (torch.optim.Optimizer): optimizer
    '''
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=momentum)
    else:
        print("Automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs)

    return optimizer, lr_scheduler


def train(model, train_loader, optimizer, criterion, device, scaler=None, scheduler=None):
    # Train the model
    model.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    try:
        n_classes = len(train_loader.dataset.classes)
    except:
        n_classes = 10

    for frame, label in tqdm(train_loader):
        optimizer.zero_grad()
        frame = frame.to(device)
        frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
        label = label.to(device)
        # If label is not one-hot,
        if len(label.shape) == 1:
            label = F.one_hot(label, n_classes).float()
        # Rember to have the float() here
        # label = F.one_hot(label, 11).float()
        if scaler is not None:
            with amp.autocast():
                # Mean is important; (https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based_en/conv_fashion_mnist.html)
                # we need to average the output in the time-step dimension to get the firing rates,
                # and then calculate the loss and accuracy by the firing rates
                out_fr = model(frame).mean(0)
                loss = criterion(out_fr, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out_fr = model(frame).mean(0)
            loss = criterion(out_fr, label)
            loss.backward()
            optimizer.step()

        label = label.argmax(1)
        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out_fr.argmax(1) == label).float().sum().item()

        functional.reset_net(model)

    train_loss /= train_samples
    train_acc /= train_samples

    if scheduler is not None:
        scheduler.step()

    return train_loss, train_acc


def train_adaptive(model, clean_trainloader, bk_trainloader, optimizer, criterion, device, scaler=None, scheduler=None, epsilon=0.1, K=0.1):

    model.train()
    train_loss = 0
    train_acc = 0
    train_samples = 0
    try:
        n_classes = len(clean_trainloader.dataset.classes)
    except:
        n_classes = 10

    # Calculate the number of bk batches. That is the total number of batches * epsilon
    bk_batches = int(len(clean_trainloader) * epsilon)

    # Now get the random indexes of bk batches
    bk_indexes = torch.randperm(len(clean_trainloader))[:bk_batches]

    print(f' Number of bk batches: {bk_batches} / {len(clean_trainloader)}')
    i = 0
    for clean, bk in tqdm(zip(clean_trainloader, bk_trainloader)):
        optimizer.zero_grad()

        clean_frame, clean_label = clean
        bk_frame, bk_label = bk
        clean_frame = clean_frame.to(device)
        clean_frame = clean_frame.transpose(0, 1)

        clean_label = clean_label.to(device)
        # If label is not one-hot,
        if len(clean_label.shape) == 1:
            clean_label = F.one_hot(clean_label, n_classes).float()

        if i in bk_indexes:
            bk_frame = bk_frame.to(device)
            bk_frame = bk_frame.transpose(0, 1)

            bk_label = bk_label.to(device)
            # If label is not one-hot,
            if len(bk_label.shape) == 1:
                bk_label = F.one_hot(bk_label, n_classes).float()

            latent_bk, out_bk = model(bk_frame)
            out = out_bk.mean(0)
            loss_bk = criterion(out, bk_label)
            functional.reset_net(model)

            latent_clean, out_clean = model(clean_frame)
            out_clean = out_clean.mean(0)
            loss_clean = criterion(out_clean, clean_label)

            # Calculate the latent distance using MSE
            # Maybe here we should consider using;
            # latent_distance = F.cosine_similarity(K * latent_bk, latent_clean)
            # loss = loss_bk + (lamda * latent_distance) + loss_clean
            # I also think that we could create another adaptive apprioach. Since FP check for the high activation neurons
            # we could take a pretrained model, and then retrain ensuring that all the neuron in the las conv layer (which is the one used by FP)
            # have the same activation value. Way may ensure that we are only changing the neurons of the last conv layer. while freezing the rest

            latent_distance = F.mse_loss(latent_bk, latent_clean)
            loss = loss_bk + (K * latent_distance) + loss_clean
            label = bk_label.argmax(1)
        else:
            # Maybe this has to be done without gradients. But I dont know yet
            latent_clean, out_clean = model(clean_frame)
            out = out_clean.mean(0)
            loss = criterion(out, clean_label)
            label = clean_label.argmax(1)

        loss.backward()
        optimizer.step()

        # label = label.argmax(1)
        train_samples += label.numel()
        train_loss += loss.item() * label.numel()
        train_acc += (out.argmax(1) == label).float().sum().item()

        functional.reset_net(model)
        i += 1

    train_loss /= train_samples
    train_acc /= train_samples

    if scheduler is not None:
        scheduler.step()

    return train_loss, train_acc


def evaluate_adaptive(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for frame, label in tqdm(test_loader):
            frame = frame.to(device)
            # [N, T, C, H, W] -> [T, N, C, H, W]
            frame = frame.transpose(0, 1)
            label = label.to(device)
            # label_onehot = F.one_hot(label, 11).float()
            _, out_fr = model(frame)
            out_fr = out_fr.mean(0)
            loss = criterion(out_fr, label)

            label = label.argmax(1)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(model)

    test_loss /= test_samples
    test_acc /= test_samples

    return test_loss, test_acc


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for frame, label in tqdm(test_loader):
            frame = frame.to(device)
            # [N, T, C, H, W] -> [T, N, C, H, W]
            frame = frame.transpose(0, 1)
            label = label.to(device)
            # label_onehot = F.one_hot(label, 11).float()
            out_fr = model(frame).mean(0)
            loss = criterion(out_fr, label)

            label = label.argmax(1)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(model)

    test_loss /= test_samples
    test_acc /= test_samples

    return test_loss, test_acc


def path_name(args):
    """
    Generate the path name based on th experiment arguments. Use a function for
    that to allow checking the existence of the path from different scripts.
    Parameters:
        args (argparse.Namespace): script arguments.
    Returns:
        path (string): The path used to save our experiments
    """
    if args.epsilon == 0.0:
        path = f'clean_{args.dataset}_{args.seed}'
    elif args.type == 'smart' or args.type == 'dynamic':
        path = f'{args.dataset}_{args.type}_{args.epsilon}_{args.trigger_size}_{args.seed}'
    else:
        path = f'{args.dataset}_{args.type}_{args.epsilon}_{args.trigger_size}_{args.pos}_{args.polarity}_{args.seed}'

    path = os.path.join(args.save_path, path)
    return path


def backdoor_model_trainer(model, criterion, optimizer, epochs, poison_trainloader, clean_testloader,
                           poison_testloader, device, scaler=None, scheduler=None):

    list_train_loss = []
    list_train_acc = []
    list_test_loss = []
    list_test_acc = []
    list_test_loss_backdoor = []
    list_test_acc_backdoor = []

    print(f'\n[!] Training the model for {epochs} epochs')
    print(f'\n[!] Trainset size is {len(poison_trainloader.dataset)},'
          f'Testset size is {len(clean_testloader.dataset)},'
          f'and the poisoned testset size is {len(poison_testloader.dataset)}'
          )

    for epoch in range(epochs):

        train_loss, train_acc = train(
            model, poison_trainloader, optimizer, criterion, device, scaler, scheduler)

        test_loss_clean, test_acc_clean = evaluate(
            model, clean_testloader, criterion, device)

        test_loss_backdoor, test_acc_backdoor = evaluate(
            model, poison_testloader, criterion, device)

        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        list_test_loss.append(test_loss_clean)
        list_test_acc.append(test_acc_clean)
        list_test_loss_backdoor.append(test_loss_backdoor)
        list_test_acc_backdoor.append(test_acc_backdoor)

        print(f'\n[!] Epoch {epoch + 1}/{epochs} '
              f'Train loss: {train_loss:.4f} '
              f'Train acc: {train_acc:.4f} '
              f'Test acc: {test_acc_clean:.4f} '
              f'Test acc backdoor: {test_acc_backdoor:.4f}'
              )

    return list_train_loss, list_train_acc, list_test_loss, list_test_acc, list_test_loss_backdoor, list_test_acc_backdoor


def backdoor_model_trainer_adaptive(model, criterion, optimizer, epochs, clean_trainloader, poison_trainloader, clean_testloader,
                                    poison_testloader, device, scaler=None, scheduler=None, epsilon=0.1, K=0.1):

    list_train_loss = []
    list_train_acc = []
    list_test_loss = []
    list_test_acc = []
    list_test_loss_backdoor = []
    list_test_acc_backdoor = []

    print(f'\n[!] Training the model for {epochs} epochs')
    print(f'\n[!] Trainset size is {len(poison_trainloader.dataset)},'
          f'Testset size is {len(clean_testloader.dataset)},'
          f'and the poisoned testset size is {len(poison_testloader.dataset)}'
          )

    for epoch in range(epochs):

        train_loss, train_acc = train_adaptive(
            model, clean_trainloader, poison_trainloader, optimizer, criterion, device, scaler, scheduler, epsilon, K)

        test_loss_clean, test_acc_clean = evaluate_adaptive(
            model, clean_testloader, criterion, device)

        test_loss_backdoor, test_acc_backdoor = evaluate_adaptive(
            model, poison_testloader, criterion, device)

        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        list_test_loss.append(test_loss_clean)
        list_test_acc.append(test_acc_clean)
        list_test_loss_backdoor.append(test_loss_backdoor)
        list_test_acc_backdoor.append(test_acc_backdoor)

        print(f'\n[!] Epoch {epoch + 1}/{epochs} '
              f'Train loss: {train_loss:.4f} '
              f'Train acc: {train_acc:.4f} '
              f'Test acc: {test_acc_clean:.4f} '
              f'Test acc backdoor: {test_acc_backdoor:.4f}'
              )

    return list_train_loss, list_train_acc, list_test_loss, list_test_acc, list_test_loss_backdoor, list_test_acc_backdoor


def plot_accuracy_combined(name, list_train_acc, list_test_acc, list_test_acc_backdoor):
    '''
    Plot the accuracy of the model in the main and backdoor test set
    Parameters:
        name (str): name of the figure
        list_train_acc (list): list of train accuracy for each epoch
        list_test_acc (list): list of test accuracy for each epoch
        list_test_acc_backdoor (list): list of test accuracy for poisoned test dataset
    Returns:
        None
    '''

    sns.set()

    fig, ax = plt.subplots(3, 1)
    fig.suptitle(name)

    ax[0].set_title('Training accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].plot(list_train_acc)

    ax[1].set_title('Test accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].plot(list_test_acc)

    ax[2].set_title('Test accuracy backdoor')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Accuracy')
    ax[2].plot(list_test_acc_backdoor)

    plt.savefig(f'{name}/accuracy.png',  bbox_inches='tight')
    # Also saving as pdf for using the plot in the paper
    plt.savefig(f'{name}/accuracy.pdf',  bbox_inches='tight')


def save_experiments(args, train_acc, train_loss, test_acc_clean, test_loss_clean, test_acc_backdoor,
                     test_loss_backdoor, model):

    # Create a folder for the experiments, by default named 'experiments'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Create if not exists a csv file, appending the new info
    path = '{}/experiments_2023_polarity.csv'.format(args.save_path)
    header = ['dataset', 'least', 'most_polarity', 'seed', 'epsilon', 'pos',
              'polarity', 'trigger_size', 'trigger_label',
              'loss', 'optimizer', 'batch_size', 'type', 'epochs',
              'train_acc', 'test_acc_clean', 'test_acc_backdoor']

    if not os.path.exists(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Append the new info to the csv file
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.dataset, args.least, args.most_polarity, args.seed, args.epsilon, args.pos,
                         args.polarity, args.trigger_size, args.trigger_label,
                         train_loss[-1], args.optim, args.batch_size, args.type, args.epochs,
                         train_acc[-1], test_acc_clean[-1], test_acc_backdoor[-1]])

    # Create a folder for the experiment, named after the experiment
    path = path_name(args)
    #path = f'experiments/{args.dataname}_{args.model}_{args.epsilon}_{args.pos}_{args.shape}_{args.trigger_size}_{args.trigger_label}'
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the info in a file
    with open(f'{path}/args.txt', 'w') as f:
        f.write(str(args))

    torch.save({
        'args': args,
        'list_train_loss': train_loss,
        'list_train_acc': train_acc,
        'list_test_loss': test_loss_clean,
        'list_test_acc': test_acc_clean,
        'list_test_loss_backdoor': test_loss_backdoor,
        'list_test_acc_backdoor': test_acc_backdoor,
    }, f'{path}/data.pt')

    torch.save(model, f'{path}/model.pth')

    plot_accuracy_combined(path, train_acc,
                           test_acc_clean, test_acc_backdoor)
    print('[!] Model and results saved successfully!')


def save_experiments_adaptive(args, train_acc, train_loss, test_acc_clean, test_loss_clean, test_acc_backdoor,
                              test_loss_backdoor, model):

    # Create a folder for the experiments, by default named 'experiments'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Create if not exists a csv file, appending the new info
    path = '{}/adaptive_polarity.csv'.format(args.save_path)
    header = ['dataset', 'least', 'most_polarity', 'seed', 'epsilon', 'K', 'pos',
              'polarity', 'trigger_size', 'trigger_label',
              'loss', 'optimizer', 'batch_size', 'type', 'epochs',
              'train_acc', 'test_acc_clean', 'test_acc_backdoor']

    if not os.path.exists(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Append the new info to the csv file
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.dataset, args.least, args.most_polarity, args.seed, args.epsilon, args.K, args.pos,
                         args.polarity, args.trigger_size, args.trigger_label,
                         train_loss[-1], args.optim, args.batch_size, args.type, args.epochs,
                         train_acc[-1], test_acc_clean[-1], test_acc_backdoor[-1]])

    # Create a folder for the experiment, named after the experiment
    path = path_name(args)
    #path = f'experiments/{args.dataname}_{args.model}_{args.epsilon}_{args.pos}_{args.shape}_{args.trigger_size}_{args.trigger_label}'

    path = os.path.join(str(args.K), path)

    if not os.path.exists(path):
        os.makedirs(path)

    # Save the info in a file
    with open(f'{path}/args.txt', 'w') as f:
        f.write(str(args))

    torch.save({
        'args': args,
        'list_train_loss': train_loss,
        'list_train_acc': train_acc,
        'list_test_loss': test_loss_clean,
        'list_test_acc': test_acc_clean,
        'list_test_loss_backdoor': test_loss_backdoor,
        'list_test_acc_backdoor': test_acc_backdoor,
    }, f'{path}/data.pt')

    torch.save(model, f'{path}/model.pth')

    plot_accuracy_combined(path, train_acc,
                           test_acc_clean, test_acc_backdoor)
    print('[!] Model and results saved successfully!')
