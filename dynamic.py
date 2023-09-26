import torch
import argparse
import numpy as np
from models import get_model, Autoencoder, AutoencoderMNIST, AutoencoderCaltech
from utils import loss_picker, optimizer_picker
from datasets import get_dataset
from spikingjelly.activation_based import functional, neuron
from tqdm import tqdm
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, neuron
from spikingjelly.datasets import play_frame
import os
import csv
import random
import cupy
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='gesture', help='Dataset to use')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--train_epochs', type=int,
                    default=1, help='Number of epochs')
parser.add_argument('--T', default=16, type=int,
                    help='simulating time-steps')
parser.add_argument('--amp', action='store_true',
                    help='Use automatic mixed precision training')
parser.add_argument('--cupy', action='store_true', help='Use cupy')
parser.add_argument('--loss', type=str, default='mse',
                    help='Loss function', choices=['mse', 'cross'])
parser.add_argument('--optim', type=str, default='adam',
                    help='Optimizer', choices=['adam', 'sgd'])
parser
# Trigger related parameters
parser.add_argument('--trigger_label', default=0, type=int,
                    help='The index of the trigger label')

parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--alpha', default=0.5, type=float, help='Alpha')
parser.add_argument('--beta', default=0.1, type=float,
                    help='Beta. Gamma in the paper')
# Other
parser.add_argument('--data_dir', type=str,
                    default='data', help='Data directory')
parser.add_argument('--save_path', type=str,
                    default='experiments', help='Path to save the experiments')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()


def clear_grad(model):
    # This is more optimal than using .zero_grad() (https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
    for p in model.parameters():
        p.grad = None


def clip_image(image, noise, eps):
    '''
    Clip the noise so its l infinity norm is less than eps
    noise shape: [T, N, C, H, W]
    image shape: [T, N, C, H, W]
    '''
    noise = noise * eps
    return noise + image


def train(args, atkmodel, tgtmodel, clsmodel, device, train_loader,
          tgtoptimizer, clsoptimizer, criterion, epoch, path_figs):
    clsmodel.train()
    atkmodel.eval()
    tgtmodel.train()

    clsmodel = clsmodel.to(device)
    atkmodel = atkmodel.to(device)
    tgtmodel = tgtmodel.to(device)

    try:
        n_classes = len(train_loader.dataset.classes)
    except:
        n_classes = 10

    crop = None
    if args.dataset == 'caltech':
        n_classes = 101
        crop = transforms.CenterCrop((180, 180))

    bk_label_one_hot = F.one_hot(torch.tensor(
        args.trigger_label).long(), n_classes).float()

    loss_list = []
    for frame, label in tqdm(train_loader):
        tmpmodel = get_model(args.dataset, args.T).to(device)
        functional.set_step_mode(tmpmodel, 'm')
        if args.cupy:
            functional.set_backend(tmpmodel, 'cupy', instance=neuron.LIFNode)

        frame = frame.to(device)
        frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
        label = label.to(device)
        bk_label = bk_label_one_hot.repeat(len(label), 1).to(device)
        label = F.one_hot(label, n_classes).float()

        if crop is not None:
            frame = crop(frame)

        # Create bk data
        noise = tgtmodel(frame)
        functional.reset_net(tgtmodel)

        atkdata = clip_image(frame, noise, args.beta)

        # This updates the weights of the autoencoder
        output_bk = clsmodel(atkdata).mean(0)
        functional.reset_net(clsmodel)

        output = clsmodel(frame).mean(0)
        functional.reset_net(clsmodel)

        loss_clean = criterion(output, label)
        bk_loss = criterion(output_bk, bk_label)
        loss = loss_clean * args.alpha + bk_loss * (1 - args.alpha)
        loss_list.append(loss.item())
        clear_grad(clsmodel)

        paragrads = torch.autograd.grad(loss, clsmodel.parameters(),
                                        create_graph=True)
        for i, (layername, layer) in enumerate(clsmodel.named_parameters()):
            modulenames, weightname = \
                layername.split('.')[:-1], layername.split('.')[-1]
            module = tmpmodel._modules[modulenames[0]]
            # TODO: could be potentially faster if we save the intermediate mappings
            for name in modulenames[1:]:
                module = module._modules[name]

            module._parameters[weightname] = \
                layer - clsoptimizer.param_groups[0]['lr'] * paragrads[i]

        tgtoptimizer.zero_grad()

        noise = tgtmodel(frame)
        atkdata = clip_image(frame, noise, args.beta)
        output = tmpmodel(atkdata).mean(0)
        loss2 = criterion(output, bk_label)
        loss2.backward()
        tgtoptimizer.step()
        functional.reset_net(tgtmodel)

        with torch.no_grad():
            noise = atkmodel(frame)
            functional.reset_net(atkmodel)

        atkdata = clip_image(frame, noise, args.beta)
        bk_output = clsmodel(atkdata).mean(0)
        functional.reset_net(clsmodel)

        clean_output = clsmodel(frame).mean(0)
        functional.reset_net(clsmodel)

        bk_loss = criterion(bk_output, bk_label)
        clean_loss = criterion(clean_output, label)
        loss = clean_loss * args.alpha + bk_loss * (1 - args.alpha)
        clsoptimizer.zero_grad()
        loss.backward()
        clsoptimizer.step()

    # Save some frames
    x = noise[:, 0, :, :, :].clone().detach() * args.beta
    play_frame(x, f'{path_figs}/{epoch}_noise_projection.gif')
    x = noise[:, 0, :, :, :].clone().detach()
    play_frame(x, f'{path_figs}/{epoch}_noise.gif')
    x_bk = atkdata[:, 0, :, :, :].clone().detach()
    play_frame(x_bk, f'{path_figs}/{epoch}_backdoor_bk.gif')
    x_clean = frame[:, 0, :, :, :].clone().detach()
    play_frame(x_clean, f'{path_figs}/{epoch}_backdoor_clean.gif')

    return sum(loss_list) / len(loss_list)


def test(args, atkmodel, scratchmodel, device,
         train_loader, test_loader, criterion, trainepoch):
    '''
    It is important to note that for testing we need to train a model from scratch.
    Note that the autoencoder has been trained simultaneously with the classifier,
    so we need to train a new classifier from scratch in order to test the performance of the autoencoder.
    Otherwise the results are not fair. This is something previously done in LIRA and DeepConfuse.
    '''
    test_loss = 0
    test_acc = 0
    test_bk_loss = 0
    test_bk_acc = 0
    atkmodel.eval()
    scratchmodel.train()
    testoptimizer, _ = optimizer_picker(args.optim, scratchmodel.parameters(),
                                        lr=args.lr, momentum=args.momentum, epochs=args.epochs)

    try:
        n_classes = len(train_loader.dataset.classes)
    except:
        n_classes = 10

    crop = None
    if args.dataset == 'caltech':
        n_classes = 101
        crop = transforms.CenterCrop((180, 180))

    bk_label_one_hot = F.one_hot(torch.tensor(
        args.trigger_label).long(), n_classes).float()

    # Train the model from scratch
    for i in range(trainepoch):
        for batch_idx, (frame, label) in enumerate(train_loader):
            frame = frame.to(device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(device)
            bk_label = bk_label_one_hot.repeat(len(label), 1).to(device)
            label = F.one_hot(label, n_classes).float()

            if crop is not None:
                frame = crop(frame)

            testoptimizer.zero_grad()
            with torch.no_grad():
                noise = atkmodel(frame)
                atkdata = clip_image(frame, noise, args.beta)
                functional.reset_net(atkmodel)

            clean_output = scratchmodel(frame).mean(0)
            functional.reset_net(scratchmodel)

            bk_output = scratchmodel(atkdata).mean(0)
            clean_loss = criterion(clean_output, label)
            bk_loss = criterion(bk_output, bk_label)

            loss = clean_loss * args.alpha + bk_loss * (1 - args.alpha)
            loss.backward()
            testoptimizer.step()
            functional.reset_net(scratchmodel)

    # Test the model from scratch
    test_samples = 0
    scratchmodel.eval()
    with torch.no_grad():
        for batch_idx, (frame, label) in enumerate(test_loader):
            frame = frame.to(device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(device)
            bk_label = bk_label_one_hot.repeat(len(label), 1).to(device)
            label = F.one_hot(label, n_classes).float()

            if crop is not None:
                frame = crop(frame)

            output = scratchmodel(frame).mean(0)
            loss = criterion(output, label)
            label = label.argmax(1)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (output.argmax(1) == label).float().sum().item()
            functional.reset_net(scratchmodel)

            noise = atkmodel(frame)
            atkdata = clip_image(frame, noise, args.beta)
            # Backdoor accuracy will go high if we use torch.clamp(frame + noise, 0, 1)
            # Even if we set beta to 0

            bk_output = scratchmodel(atkdata).mean(0)
            loss = criterion(bk_output, bk_label)

            test_bk_loss += loss.item() * label.numel()
            test_bk_acc += (bk_output.argmax(1) ==
                            args.trigger_label).float().sum().item()
            functional.reset_net(atkmodel)
            functional.reset_net(scratchmodel)

    test_loss /= test_samples
    test_bk_loss /= test_samples

    test_acc /= test_samples
    test_bk_acc /= test_samples

    return test_loss, test_acc, test_bk_loss, test_bk_acc


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0

    try:
        n_classes = len(test_loader.dataset.classes)
    except:
        n_classes = 10

    crop = None
    if args.dataset == 'caltech':
        n_classes = 101
        crop = transforms.CenterCrop((180, 180))

    with torch.no_grad():
        for frame, label in tqdm(test_loader):
            frame = frame.to(device)
            # [N, T, C, H, W] -> [T, N, C, H, W]
            frame = frame.transpose(0, 1)
            label = label.to(device)
            label = F.one_hot(label, n_classes).float()

            if crop is not None:
                frame = crop(frame)

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


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    path_dataset = os.path.join(args.save_path, args.dataset)
    if not os.path.exists(path_dataset):
        os.makedirs(path_dataset)

    exp_path = os.path.join(path_dataset, f'{str(args)}')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    path_figs = os.path.join(exp_path, 'figs')
    if not os.path.exists(path_figs):
        os.makedirs(path_figs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    clsmodel = get_model(args.dataset, args.T)
    if args.dataset == 'mnist':
        atkmodel = AutoencoderMNIST(spiking_neuron=neuron.IFNode,
                                    surrogate_function=surrogate.ATan(),  detach_reset=True)
        tgtmodel = AutoencoderMNIST(spiking_neuron=neuron.IFNode,
                                    surrogate_function=surrogate.ATan(),  detach_reset=True)

    elif args.dataset == 'caltech':
        atkmodel = AutoencoderCaltech(spiking_neuron=neuron.IFNode,
                                      surrogate_function=surrogate.ATan(),  detach_reset=True)
        tgtmodel = AutoencoderCaltech(spiking_neuron=neuron.IFNode,
                                      surrogate_function=surrogate.ATan(),  detach_reset=True)

    else:
        atkmodel = Autoencoder(spiking_neuron=neuron.IFNode,
                               surrogate_function=surrogate.ATan(),  detach_reset=True)
        tgtmodel = Autoencoder(spiking_neuron=neuron.IFNode,
                               surrogate_function=surrogate.ATan(),  detach_reset=True)

    tgtmodel.load_state_dict(atkmodel.state_dict())
    tgtoptimizer, _ = optimizer_picker(
        args.optim, tgtmodel.parameters(), args.lr, args.momentum, args.epochs)

    functional.set_step_mode(clsmodel, 'm')
    functional.set_step_mode(atkmodel, 'm')
    functional.set_step_mode(tgtmodel, 'm')

    if args.cupy:
        functional.set_backend(clsmodel, 'cupy', instance=neuron.LIFNode)
        functional.set_backend(atkmodel, 'cupy', instance=neuron.LIFNode)
        functional.set_backend(tgtmodel, 'cupy', instance=neuron.LIFNode)
        cupy.random.seed(args.seed)

    # Load the dataset
    train_data, test_data = get_dataset(args.dataset, args.T, args.data_dir)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=1)

    best_bk_acc = 0
    best_epoch = 0
    best_clean_acc = 0
    train_loss = 0
    scratchmodel = None
    criterion = loss_picker(args.loss)
    list_bk = []
    list_clean = []
    list_ae = []
    list_cls = []
    for epoch in range(1, args.epochs + 1):
        for i in range(args.train_epochs):
            clsoptimizer, _ = optimizer_picker(
                args.optim, clsmodel.parameters(), args.lr, args.momentum, args.epochs)
            train_loss = train(args, atkmodel, tgtmodel, clsmodel, device, train_loader,
                               tgtoptimizer, clsoptimizer, criterion, epoch, path_figs)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(train_loader.dataset),
                len(train_loader.dataset) * args.epochs,
                100. * i / len(train_loader), train_loss))
        atkmodel.load_state_dict(tgtmodel.state_dict())
        # Remove scratchmodel to save memory
        if scratchmodel is not None:
            del scratchmodel

        scratchmodel = get_model(args.dataset, args.T).to(device)
        scratchmodel.load_state_dict(clsmodel.state_dict())
        functional.set_step_mode(scratchmodel, 'm')
        if args.cupy:
            functional.set_backend(scratchmodel, 'cupy',
                                   instance=neuron.LIFNode)

        test_loss_clean, test_acc_clean, test_bk_loss, test_bk_acc = test(args, atkmodel, scratchmodel, device,
                                                                          train_loader, test_loader, criterion, trainepoch=args.train_epochs)

        list_bk.append(test_bk_acc)
        list_clean.append(test_acc_clean)
        list_ae.append(atkmodel.state_dict())
        list_cls.append(clsmodel.state_dict())

        # Print test clean and backdoor accuracy
        print('Test clean accuracy: {:.2f}%, Test backdoor accuracy: {:.2f}%'.format(
            test_acc_clean * 100, test_bk_acc * 100))

    # This would be better if we put inside the training loop so we dont store all the models
    for epoch, (a, b) in enumerate(zip(list_clean, list_bk)):
        if a > best_clean_acc and b > best_bk_acc:
            best_clean_acc = a
            best_bk_acc = b
            best_epoch = epoch
            best_ae = list_ae[epoch]
            best_cls = list_cls[epoch]

    print('Best clean accuracy: {:.2f}%, Best backdoor accuracy: {:.2f}%, Best epoch: {}'.format(
        best_clean_acc * 100, best_bk_acc * 100, best_epoch))

    torch.save({
        'ae_state_dict': best_ae,
        'model_state_dict': best_cls,
        'best_clean_acc': best_clean_acc,
        'best_bk_acc': best_bk_acc,
        'best_epoch': best_epoch,
        'train_loss': train_loss,
        'args': args},
        f'{exp_path}/args.pth')

    # Create if not exists a csv file, appending the new info
    path = '{}/dynamic_experiments.csv'.format(args.save_path)
    header = ['dataset', 'seed', 'alpha', 'beta', 'epochs', 'trigger_label',
              'loss', 'optimizer', 'batch_size', 'train_epochs',
              'test_acc_clean', 'test_acc_backdoor', 'best_epoch']

    if not os.path.exists(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Append the new info to the csv file
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.dataset, args.seed, args.alpha, args.beta,
                         args.epochs, args.trigger_label, args.loss,
                         args.optim, args.batch_size, args.train_epochs,
                         best_clean_acc, best_bk_acc, best_epoch])


if __name__ == '__main__':
    main()
