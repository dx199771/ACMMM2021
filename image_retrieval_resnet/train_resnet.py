import argparse
import os
import pickle
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm
from typing import Type, Any, Callable, Union, List, Optional
from utils.general import check_file
from utils.datasets import create_dataloader
from utils.utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser()

    model_names = sorted(name for name in models.resnet.__dict__
                         if name.islower() and not name.startswith("__")
                         and name.startswith("resnet")
                         and callable(models.resnet.__dict__[name]))

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: resnet50)')
    parser.add_argument('--pretrained', dest='pretrained', default=False, action='store_true', help='use pre-trained model')
    parser.add_argument('--weights', type=str, default='resnet.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default='data/image_objects.yaml', help='data.yaml path')
    parser.add_argument('--image-size', type=int, default=224, help='resize to this image size')
    parser.add_argument('--cache-image', type=bool, default=True, help='cache image into memory')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('-lr', '--learning-rate', default=0.05, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--evaluate', default=False, type=bool, help='evaluate only')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--save-dir', dest='save_dir', help='The directory used to save the trained models',
                        default='save_temp', type=str)

    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = models.resnet.__dict__[args.arch](pretrained=args.pretrained)

    model = model.to(device)

    args.start_epoch = 0
    best_prec1 = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Load data
    args.data = check_file(args.data)
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, args.data)  # check

    transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor(),
            normalize,
        ])
    # Trainloader
    train_loader, dataset = create_dataloader(train_path, args.batch_size, cache=args.cache_image, transform=transform)
    dataset.classes = names

    val_loader = create_dataloader(test_path, args.batch_size, cache=False, transform=transform)[0]

    #  Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        top1_accuracy, top5_accuracy = validate(val_loader, model)
        print("Evaluate result: accuracy on validation set:  %f%%(top1), %f%%(top5)" % (
            top1_accuracy, top5_accuracy))
        return

    train_hist = {}
    train_hist['losses'] = []
    train_hist['top1_accuracies'] = []
    train_hist['top5_accuracies'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, args.learning_rate, epoch)

        # train for one epoch
        # train for one epoch
        epoch_start_time = time.time()
        epoch_loss, _ = train(train_loader, model, criterion, optimizer, epoch)
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        # evaluate on validation set
        top1_accuracy, top5_accuracy = validate(val_loader, model)
        print("Epoch %d of %d with %.2f s" % (epoch + 1, args.epochs, per_epoch_ptime))
        print("loss: %.8f， accuracy on validation set:  %f%%(top1), %f%%(top5)" % (
        epoch_loss, top1_accuracy, top5_accuracy))

        # remember best prec@1 and save checkpoint
        is_best = top1_accuracy > best_prec1
        best_prec1 = max(top1_accuracy, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.tar'))

        train_hist['losses'].append(epoch_loss)
        train_hist['top1_accuracies'].append(top1_accuracy)
        train_hist['top5_accuracies'].append(top5_accuracy)
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (
        np.mean(train_hist['per_epoch_ptimes']), args.epochs, total_ptime))
    print("Training finish!... save training results")

    # Save training hist
    with open(os.path.join(args.save_dir, args.arch + '_train_hist.pkl'), 'wb') as f:
        pickle.dump(train_hist, f)
    # Show training hist and save
    title = "Training Model: %s" % (args.arch)
    savePath = os.path.join(args.save_dir, args.arch + '_train_hist.png')
    show_train_hist(train_hist, title, save=True, path=savePath)

    # Save model
    torch.save(model.state_dict(), os.path.join(args.save_dir, args.arch + ".pth"))
    print("Model saved!")


def adjust_learning_rate(optimizer, initial_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 20 epochs"""
    lr = initial_lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()
    Loss = []
    top1 = AverageMeter()
    for (input, target) in tqdm(train_loader, file=sys.stdout):
        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Loss.append(loss.item())

        # Get the class of top1
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target)
        batch_accuracy = correct[:].view(-1).float().sum(0).mul_(100.0 / batch_size)
        top1.update(batch_accuracy, batch_size)

    # Return mean loss
    epoch_loss = np.mean(Loss)
    return epoch_loss, top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


def validate(test_loader, model):
    # switch to evaluate mode
    model.eval()

    top1 = AverageMeter()
    top5 = AverageMeter()

    for (input, target) in tqdm(test_loader, file=sys.stdout):
        input = input.to(device)
        target = target.to(device)
        batch_size = target.size(0)
        # compute output
        with torch.no_grad():
            output = model(input)

        # Get the class of top1
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        batch_accuracy = correct[:1].view(-1).float().sum(0).mul_(100.0 / batch_size)
        top1.update(batch_accuracy, batch_size)

        # Get the class of top5
        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        batch_accuracy = correct[:5].contiguous().view(-1).float().sum(0).mul_(100.0 / batch_size)
        top5.update(batch_accuracy, batch_size)

    # Return accuracy
    return top1.avg, top5.avg


# Show training histogram
def show_train_hist(hist, title, show=False, save=False, path='Train_hist.png'):
    """Loss and accuracy tracker

    Plot the losses of generator and discriminator independently to see the trend

    Arguments:
        hist {[dict]} -- Tracking variables

    Keyword Arguments:
        show {bool} -- If to display the figure (default: {False})
        save {bool} -- If to store the figure (default: {False})
        path {str} -- path to store the figure (default: {'Train_hist.png'})
    """
    x = range(len(hist['losses']))
    losses = hist['losses']
    top1_accuracies = hist['top1_accuracies']
    top5_accuracies = hist['top5_accuracies']

    plt.figure(figsize=(16, 8))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.plot(x, losses)
    plt.title('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x, top1_accuracies, label="top1")
    plt.plot(x, top5_accuracies, label="top5")
    plt.legend(loc=4)
    plt.title('Accuracy on validation set')
    plt.grid(True)

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_dataset(dataset, channels, dim=(5, 5), path='result.png'):
    figure = plt.figure(figsize=(8, 8))
    cols = dim[0]
    rows = dim[1]
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(dataset.classes[label])
        plt.axis("off")
        if channels == 1:
            plt.imshow(img.squeeze())
        else:
            # rearrange the order
            plt.imshow(img.squeeze().permute(1,2,0))
    plt.savefig(path)

if __name__ == '__main__':
    main()

