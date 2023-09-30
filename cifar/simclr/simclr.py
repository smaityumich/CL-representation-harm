import os
path = 'path/to/folder'
os.chdir(path)

import numpy as np
from PIL import Image
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms
from models import SimCLR
import sys, json
from sklearn.linear_model import LogisticRegression
from warnings import filterwarnings
filterwarnings('ignore')




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
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


class CIFAR10Pair(CIFAR10):
    
    """Generate mini-batche pairs on CIFAR10 training set."""
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # stack a positive pair


def nt_xent(x, t=0.5):
    x = F.normalize(x, dim=1)
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def get_train_loader(undersample_class, batch_size = 512, workers = 16):
    data_dir = 'data/'
    if undersample_class < 10:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          get_color_distortion(s=0.5),
                                          transforms.ToTensor()])
        train_set = CIFAR10Pair(root=data_dir, train=True, transform=train_transform, download=True)
        N = 100 # N to 1 downsampling
        targets = np.array(train_set.targets)
        index = []
        for i in range(10):
            if i != undersample_class:
                index += list(np.where(targets == i)[0])
            else:
                index_i = np.where(targets == i)[0]
                n = index_i.shape[0]
                n1 = int(n / N)
                random_index = np.arange(n)
                index += list(index_i[list(np.random.choice(random_index, size = (n1, ), replace = False))])
        train_set.targets = list(np.array(train_set.targets)[index])
        train_set.data = train_set.data[index]

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    
    else:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          get_color_distortion(s=0.5),
                                          transforms.ToTensor()])
        train_set = CIFAR10Pair(root=data_dir, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
        
    return train_loader

def train(undersample_class = 10, epochs = 1000, seed = 12121):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # seed options
    device = 'cuda'
    
    # data options
    dataset = 'cifar10'
    data_dir = 'data/'
    batch_size = 512
    workers = 16
    
    # network options
    backbone = 'resnet34'
    projection_dim = 128
    
    assert undersample_class in range(11)

    # loss options
    optimizer = 'sgd'
    learning_rate = 0.6 # initial lr = 0.3 * batch_size / 256
    momentum = 0.9
    weight_decay = 1.0e-6 # "optimized using LARS [...] and weight decay of 10−6"
    temperature = 0.5 # see appendix B.7.: Optimal temperature under different batch sizes
    
    assert torch.cuda.is_available()
    cudnn.benchmark = True

    # Prepare model
    assert backbone in ['resnet18', 'resnet34', 'resnet50']
    base_encoder = eval(backbone)
    model = SimCLR(base_encoder, projection_dim=projection_dim).cuda()

    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    train_loader = get_train_loader(undersample_class, batch_size = batch_size, workers = workers)

    # cosine annealing lr
    scheduler = LambdaLR(optimizer,
                    lr_lambda=lambda step: get_lr(step, epochs * len(train_loader), learning_rate, 1e-3) 
                        )
    
    # test data
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set.targets), shuffle=False, drop_last=True)
    images_train, labels_train = next(iter(data_loader))
    labels_train = labels_train.numpy()

    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    data_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set.targets), shuffle=False, drop_last=True)
    images_test, labels_test = next(iter(data_loader))
    labels_test = labels_test.numpy()
    
    

    # SimCLR training
    
    
    ret_list = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = AverageMeter("SimCLR_loss")
        for x, y in train_loader:
            sizes = x.size()
            x = x.view(sizes[0] * 2, sizes[2], sizes[3], sizes[4]).cuda(non_blocking=True)

            optimizer.zero_grad()
            feature, rep = model(x)
            loss = nt_xent(rep, temperature)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), x.size(0))
            
        print('Epoch {}/{} loss {}'.format(epoch, epochs, loss_meter.avg))
        
        if epoch % 50 == 0:
            model.eval()
            model.to(device)
            n = 500
            n_train = labels_train.shape[0]
            n_test = labels_test.shape[0]
            n_train_p, n_test_p = int(n_train / n) * n, int(n_test / n) * n
            with torch.no_grad():
                z_list, p_list = [], []
                for i in range(int(n_train / n)):
                    z, p = model(images_train[(i * n):(i * n + n)].to(device))
                    z_list.append(z.to('cpu').numpy())
                    p_list.append(p.to('cpu').numpy())
                rep_train_z, rep_train_p = np.concatenate(z_list, axis = 0), np.concatenate(p_list, axis = 0)
                z_list, p_list = [], []
                for i in range(int(n_test / n)):
                    z, p = model(images_test[(i * n):(i * n + n)].to(device))
                    z_list.append(z.to('cpu').numpy())
                    p_list.append(p.to('cpu').numpy())
                rep_test_z, rep_test_p = np.concatenate(z_list, axis = 0), np.concatenate(p_list, axis = 0)

            LogReg = LogisticRegression(C = 0.1).fit(rep_train_z, labels_train)
            acc = LogReg.score(rep_test_z, labels_test)
            ret_dict = {'backbone': backbone, 
                       'undersample': undersample_class, 
                       'epoch': epoch, 
                       'acc': acc}
            print(ret_dict)
            ret_list.append(ret_dict)
            
        if epoch % 500 == 0:
            # model_dict = {'epoch': epochs, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),}
            # torch.save(model_dict, 'models_upd/simclr_{}_epoch_{}_undersampled_{}.pt'.format(backbone, epochs, undersample_class))

            model.eval()
            model.to(device)
            n = 500
            n_train = labels_train.shape[0]
            n_test = labels_test.shape[0]
            n_train_p, n_test_p = int(n_train / n) * n, int(n_test / n) * n
            with torch.no_grad():
                z_list, p_list = [], []
                for i in range(int(n_test / n)):
                    z, p = model(images_test[(i * n):(i * n + n)].to(device))
                    z_list.append(z.to('cpu').numpy())
                    p_list.append(p.to('cpu').numpy())
                z_test, p_test = np.concatenate(z_list, axis = 0), np.concatenate(p_list, axis = 0)

            with open(path + 'features/simclr_{}_epoch_{}_undersampled_{}_seed_{}.npy'.format(backbone, epoch, undersample_class, seed), 'wb') as f:
                np.save(f, labels_test[:n_test_p])
                np.save(f, z_test)
                np.save(f, p_test)

            
    return ret_list
    
        
        

if __name__ == '__main__':
    with open(path + 'seeds.npy', 'rb') as f:
        seeds = np.load(f)
    
    idx = int(float(sys.argv[1]))
    class_idx, seed_idx = idx // 10, idx % 10
    seed = seeds[seed_idx]
    r = train(epochs = 500, undersample_class = class_idx, seed = seed)
    

