import argparse
import os
from os import path
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
from eval_cifar import eval_test
from data.memoboosted_cifar import memoboosted_CIFAR10
from data.augmentations import cifar_tfs_train, cifar_tfs_test
from models.simclr import SimCLR
from losses.nt_xent import NT_Xent_Loss
from torchvision.datasets import CIFAR10
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms
warnings.filterwarnings("ignore")

pwd = 'path/to/folder'

parser = argparse.ArgumentParser(description='PyTorch Cifar10-LT Self-supervised Training')
parser.add_argument('experiment', type=str)
parser.add_argument('--save_dir', default='checkpoints', type=str, help='path to save checkpoint')
parser.add_argument('--dataset', type=str, default='cifar10', help="dataset-cifar10")
parser.add_argument('--trainSplit', type=str, default='', help="train split")
parser.add_argument("--gpus", type=str, default="0", help="gpu id sequence split by comma")
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--num_workers', type=int, default=16, help='num workers')
parser.add_argument('--model', default='resnet34', type=str, help='model type')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=1000, type=int, help='training epochs')
parser.add_argument('--num_class', default=10, type=int, help='num class')
parser.add_argument('--print_freq', default=20, type=int, help='print frequency')
parser.add_argument('--save_freq', default=500, type=int, help='save frequency /epoch')
parser.add_argument('--eval_freq', default=20, type=int, help='eval frequency /epoch')
parser.add_argument('--checkpoint', default='', type=str, help='saving pretrained model')
parser.add_argument('--resume', default=False, type=bool, help='resume training')
parser.add_argument('--lr', default=0.6, type=float, help='optimizer lr')
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--temperature', default=0.2, type=float, help='nt_xent temperature')
parser.add_argument('--bcl', action='store_true', help='boosted contrastive learning')
parser.add_argument('--momentum_loss_beta', type=float, default=0.97)
parser.add_argument('--rand_k', type=int, default=1, help='k in randaugment')
parser.add_argument('--rand_strength', type=int, default=30, help='maximum strength in randaugment(0-30)')
parser.add_argument('--prune_percent', type=float, default=0, help="whole prune percentage")
parser.add_argument('--undersample_class', type = int, default = 10, help = 'class to undersample; put 10 for balanced training')
parser.add_argument('--undersample_fold', type = int, default = 100, help = 'undersampling fold')


with open(path.join(pwd, 'seeds.npy'), 'rb') as f:
    seeds = np.load(f)


def undersample(dataset, undersample_class = 10, undersample_fold = 10):
    targets = np.array(dataset.targets)
    if undersample_class >= 10:
        return list(range(targets.shape[0]))
    else:
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
        
        return index



class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.enc = base_encoder(pretrained = 'IMAGENET1K_V1')  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, projection_dim))

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection



def main():
    global args
    args = parser.parse_args()
    data_folder = 'path/to/data-folder'

    # gpu 
    gpus = list(map(lambda x: torch.device('cuda', x), [int(e) for e in args.gpus.strip().split(",")]))
    torch.cuda.set_device(gpus[0])
    torch.backends.cudnn.benchmark = True
    
    seed = seeds[args.seed]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # create model
    model = SimCLR(resnet18, projection_dim=128).cuda()

    # criterion
    criterion = NT_Xent_Loss(temp=args.temperature, average=False)

    # data augmentations
    tfs_train, tfs_test = cifar_tfs_train, cifar_tfs_test
    # loading data
    cifar = CIFAR10(data_folder, download = True)
    train_idx_list = undersample(dataset = cifar, undersample_class = args.undersample_class, undersample_fold = args.undersample_fold)
    
    if args.bcl:
        train_datasets = memoboosted_CIFAR10(train_idx_list, args, root=data_folder, train=True)
    else:
        ValueError('Only implemented for bcl')
    eval_test_datasets = torchvision.datasets.CIFAR10(root=data_folder, train=False, download=True, transform=tfs_test)
    eval_train_datasets = torchvision.datasets.CIFAR10(root=data_folder, train=True, download=True, transform=tfs_test)

    train_loader = torch.utils.data.DataLoader(train_datasets, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    eval_test_loader = torch.utils.data.DataLoader(eval_test_datasets, batch_size=1000, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    eval_train_loader = torch.utils.data.DataLoader(eval_train_datasets, batch_size=1000, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # dataset statistics
    class_stat = train_datasets.idxsNumPerClass
    dataset_total_num = np.sum(class_stat)
    print("class distribution in training set is {}".format(class_stat))

    # optimizer, training schedule
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader), 1, 1e-6 / args.lr, warmup_steps=10 * len(train_loader)))

            
    # initialize momentum loss
    shadow = torch.zeros(dataset_total_num).cuda()
    momentum_loss = torch.zeros(args.epochs,dataset_total_num).cuda()
    
    # training
    for epoch in range(1, args.epochs + 1):

        shadow, momentum_loss = train(train_loader, model, criterion, optimizer, scheduler, epoch, shadow, momentum_loss, args=args)
        if args.bcl:
            train_datasets.update_momentum_weight(momentum_loss, epoch)
     
        if (epoch) % args.eval_freq == 0:
            # linear probing on full dataset 
            acc_full = eval_test(train_loader = eval_train_loader, test_loader = eval_test_loader, model = model, epoch = epoch, args=args)
            print('epoch {}, accuracy {}'.format(epoch, acc_full))

    get_features(model, args, data_folder)
            
    
    

def train(train_loader, model, criterion, optimizer, scheduler, epoch, shadow=None, momentum_loss=None, args=None):
    losses, data_time_meter, train_time_meter = AverageMeter(), AverageMeter(), AverageMeter()
    losses.reset()
    end = time.time()

    for i, (inputs, index) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)

        scheduler.step()
        model.train()

        d = inputs.size()
        inputs = inputs.view(d[0]*2, d[2], d[3], d[4]).cuda(non_blocking=True)

        _, features = model(inputs)
        loss = criterion(features)

        for count in range(d[0]): 
            if epoch>1:
                new_average = (1.0 - args.momentum_loss_beta) * loss[count].clone().detach() + args.momentum_loss_beta * shadow[index[count]]
            else:
                new_average = loss[count].clone().detach()
                
            shadow[index[count]] = new_average
            momentum_loss[epoch-1,index[count]] = new_average

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        losses.update(float(loss.mean().detach().cpu()), inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'data_time: {data_time.val:.2f} ({data_time.avg:.2f})\t'
                     'train_time: {train_time.val:.2f} ({train_time.avg:.2f})\t'.format(
                          epoch, i, len(train_loader), loss=losses,
                          data_time=data_time_meter, train_time=train_time_meter))
        
    return shadow, momentum_loss

def get_features(model, args, data_folder):
    
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(root=data_folder, train=False, download=False, transform=transform)
    data_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set.targets), shuffle=False)
    images_test, labels_test = next(iter(data_loader))
    labels_test = labels_test.numpy()
    
    model.eval()
    n = 500
    n_test = labels_test.shape[0]
    n_test_p = int(n_test / n) * n
    with torch.no_grad():
        z_list, p_list = [], []
        for i in range(int(n_test / n)):
            z, p = model(images_test[(i * n):(i * n + n)].cuda())
            z_list.append(z.to('cpu').numpy())
            p_list.append(p.to('cpu').numpy())
        z_test, p_test = np.concatenate(z_list, axis = 0), np.concatenate(p_list, axis = 0)
    
    file = path.join(pwd, 'features/bcl_{}_epoch_{}_undersample_{}_by_{}_seed_{}.npy'.format('resnet18', args.epochs, args.undersample_class, args.undersample_fold, args.seed))
    with open(file, 'wb') as f:
        np.save(f, labels_test[:n_test_p])
        np.save(f, z_test)
        np.save(f, p_test)


if __name__ == '__main__':
    main()


