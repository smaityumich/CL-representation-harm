import argparse
import time
import math
from os import path, makedirs
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import datasets
from torchvision import transforms

from simsiam.loader import TwoCropsTransform
from simsiam.model_factory import SimSiam
from simsiam.criterion import SimSiamLoss
from simsiam.validation import KNNValidation

pwd = 'path/to/folder'

parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_root', type=str, help='path to dataset directory', default = 'path/to/data-folder')
parser.add_argument('--img_dim', default=32, type=int)
parser.add_argument('--arch', default='resnet18', help='model name is used for training')
parser.add_argument('--feat_dim', default=2048, type=int, help='feature dimension')
parser.add_argument('--num_proj_layers', type=int, default=2, help='number of projection layer')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--loss_version', default='simplified', type=str,
                    choices=['simplified', 'original'],
                    help='do the same thing but simplified version is much faster. ()')
parser.add_argument('--print_freq', default=20, type=int, help='print frequency')
parser.add_argument('--eval_freq', default=2, type=int, help='evaluate model frequency')
parser.add_argument('--seed', default = 0, type = int)

parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--undersample_class', type = int, default = 10, help = 'class to undersample; put 10 for balanced training')
parser.add_argument('--undersample_fold', type = int, default = 100, help = 'undersampling fold')

args = parser.parse_args()

def _makedir(folder):
    if not path.exists(folder):
        makedirs(folder)
    return None

_makedir(path.join(pwd, 'models/'))
_makedir(path.join(pwd, 'features/'))

with open(path.join(pwd, 'seeds.npy'), 'rb') as f:
    seeds = np.load(f)


def undersample(dataset, undersample_class = 10, undersample_fold = 10):
    if undersample_class >= 10:
        return dataset
    else:
        targets = np.array(dataset.targets)
        index = []
        for i in range(10):
            if i != undersample_class:
                index += list(np.where(targets == i)[0])
            else:
                index_i = np.where(targets == i)[0]
                n = index_i.shape[0]
                n1 = int(n / undersample_fold)
                random_index = np.arange(n)
                index += list(index_i[list(np.random.choice(random_index, size = (n1, ), replace = False))])
        dataset.targets = list(np.array(dataset.targets)[index])
        dataset.data = dataset.data[index]
        return dataset



def main():
    
    np.random.seed(seeds[args.seed])
    torch.manual_seed(seeds[args.seed])
    torch.cuda.manual_seed(seeds[args.seed])
    
    model_best = path.join(pwd, 'models/simsiam_{}_epoch_{}_undersample_{}_by_{}_seed_{}_best.pth'.format(args.arch, args.epochs, args.undersample_class, args.undersample_fold, args.seed))
    model_last = path.join(pwd, 'models/simsiam_{}_epoch_{}_undersample_{}_by_{}_seed_{}_last.pth'.format(args.arch, args.epochs, args.undersample_class, args.undersample_fold, args.seed))
    print(vars(args))

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = datasets.CIFAR10(root=args.data_root,
                                 train=True,
                                 download=True,
                                 transform=TwoCropsTransform(train_transforms))

    train_loader = DataLoader(dataset=undersample(train_set, args.undersample_class, args.undersample_fold),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    model = SimSiam(args)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    criterion = SimSiamLoss(args.loss_version)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
        cudnn.benchmark = True

    start_epoch = 1

    # routine
    best_acc = 0.0
    validation = KNNValidation(args, model.encoder)
    for epoch in range(start_epoch, args.epochs+1):

        adjust_learning_rate(optimizer, epoch, args)
        print("Training...")

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)

        if epoch % args.eval_freq == 0:
            print("Validating...")
            val_top1_acc = validation.eval()
            print('Top1: {}'.format(val_top1_acc))

            # save the best model
            if val_top1_acc > best_acc:
                best_acc = val_top1_acc

                save_checkpoint(epoch, model, optimizer, best_acc,
                                model_best,
                                'Saving the best model!')

        # # save the model
        # if epoch % args.save_freq == 0:
        #     save_checkpoint(epoch, model, optimizer, val_top1_acc,
        #                     path.join(trial_dir, 'ckpt_epoch_{}_{}.pth'.format(epoch, args.trial)),
        #                     'Saving...')

            print('epochs: {}, learning rate: {}, best accuracy: {}'.format(epoch, args.learning_rate ,best_acc))

    # save model
    save_checkpoint(epoch, model, optimizer, val_top1_acc,
                    model_last,
                    'Saving the model at the last epoch.')
    
    get_features(model, optimizer, model_best)
    
    


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        outs = model(im_aug1=images[0], im_aug2=images[1])
        loss = criterion(outs['z1'], outs['z2'], outs['p1'], outs['p2'])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(epoch, model, optimizer, acc, filename, msg):
    state = {
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'top1_acc': acc
    }
    torch.save(state, filename)
    print(msg)


def load_checkpoint(model, optimizer, filename, map_location = 'cuda:0'):
    checkpoint = torch.load(filename, map_location=map_location)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return start_epoch, model, optimizer

def get_features(model, optimizer, filename):
    start_epoch, model, optimizer = load_checkpoint(model, optimizer, filename)
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_set = datasets.CIFAR10(root=args.data_root,
                                 train=False,
                                 download=True,
                                 transform=TwoCropsTransform(test_transforms))

    test_loader = DataLoader(dataset=test_set,
                              batch_size=500,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)
    
    z_list, p_list, label_list = [], [], []
    
    model.eval()
    for images, label in test_loader:
        
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            
        with torch.no_grad():
            outs = model(im_aug1=images[0], im_aug2=images[1])
            
        z, p = outs['z1'], outs['p1']    
        z_list.append(z.to('cpu').numpy())
        p_list.append(p.to('cpu').numpy())
        label_list.append(label.numpy())
            
    z_test, p_test, labels_test = np.concatenate(z_list, axis = 0), np.concatenate(p_list, axis = 0), np.concatenate(label_list, axis = 0)
    
    file = path.join(pwd, 'features/simsiam_{}_epoch_{}_undersample_{}_by_{}_seed_{}_best.npy'.format(args.arch, args.epochs, args.undersample_class, args.undersample_fold, args.seed))
    with open(file, 'wb') as f:
        np.save(f, labels_test)
        np.save(f, z_test)
        np.save(f, p_test)


if __name__ == '__main__':
    main()



