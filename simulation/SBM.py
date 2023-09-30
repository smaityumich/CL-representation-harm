import numpy as np
import torch, os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}  \usepackage{amssymb}  \usepackage{mathrsfs}'
from graspologic.simulations import sbm
import seaborn as sns
path = "plots"
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)



def simclr(neighborhood, d = 10, lr = 0.1, tau = 1., epochs = 20000):
    
    
    assert neighborhood.shape[0] == neighborhood.shape[1]
    N = neighborhood.shape[0]
    v_init = torch.normal(0, 1, size = (N, d))
    v_init = v_init / torch.linalg.norm(v_init, dim = 1, keepdim = True)
    v = Variable(v_init, requires_grad=True)
    for i in range(epochs):
        lr_i = lr * (i + 1) ** (-0.2)
        # normalized representations
        v_normalized = v / torch.linalg.norm(v, dim = 1, keepdim = True)
        # aligned part of the loss
        degrees = neighborhood.sum(dim = 1, keepdim = True)
        nbd_means = neighborhood @ v_normalized / degrees
        loss_align = - (nbd_means * v_normalized).sum(dim = 1).mean() / tau
        # log sum exp part of the loss
        cosines = v_normalized @ v_normalized.T
        
        exp_cosines = torch.exp(cosines/tau)
        loss_unif = torch.log((exp_cosines).mean(dim = 1, keepdim = True)).mean()
        
        loss = loss_align + loss_unif
        loss.backward()
        
        # updating the weight matrix after backpropagation
        with torch.no_grad():
            v = v-(lr_i * v.grad.data)
            v = v / torch.linalg.norm(v, dim = 1, keepdim = True)
        v = Variable(v, requires_grad=True)
        
        if i % 1000 == 0: 
            print(f'Loss value at iter {i} is {loss.detach().numpy()}')
            
    v = v.detach()
    v_normalized = v / torch.linalg.norm(v, dim = 1, keepdim = True)
    
    return v_normalized.numpy()


    

if __name__ == '__main__':


    n = [256] * 3
    alpha = 0.4
    p = [[0.95, alpha, 0],
         [alpha, 0.95, 0],
         [0, 0, 0.95]
         ]
    
    cosines = []
    means = []
    stds = []
    
    for n1 in [256, 32, 4]:
        n[0] = n1
    
        np.random.seed(1)
        G = sbm(n=n, p=p)
    
    
    
        G = np.array(G, dtype = 'float32')
        v_normalized1 = simclr(neighborhood = torch.from_numpy(G), lr = 10, 
                              epochs = 5000)
        v_normalized1 = v_normalized1 / np.linalg.norm(v_normalized1, axis = 1, keepdims=True)
        cosines1 = (v_normalized1 @ v_normalized1.T)
        cosines.append(cosines1)
        
        mean1 = np.zeros((3, 3))
        std1 = np.zeros((3, 3))
        n_cumsum = [0] + list(np.cumsum(n))
        for i in range(3):
            for j in range(3):
                cosines_ij = cosines1[n_cumsum[i]:n_cumsum[i+1], n_cumsum[j]:n_cumsum[j+1]]
                mean1[i, j] = np.mean(cosines_ij)
                std1[i, j] = np.std(cosines_ij)
                
        means.append(mean1)
        stds.append(std1)
    
    
    # Activating tex in all labels globally
    plt.rc('text', usetex=True)
    # Adjust font specs as desired (here: closest similarity to seaborn standard)
    plt.rc('font', **{'size': 13.0})
    plt.rc('text.latex', preamble=r'\usepackage{lmodern}')
    
    cmap = 'Spectral'
    fig, ax = plt.subplots(1, 4, figsize = (11.25, 2.5))
    
    
    classes = [1, 2, 3]
    sns.heatmap(p, annot=True,
                        xticklabels=classes, yticklabels=classes,
                        cbar = False, ax = ax[0], cmap = cmap)
    
    ax[0].set_title(r'connectivity ($\alpha_{i, j}$)')
    
    
    mean = means[0]
    std = stds[0]
    labels = []
    for m, s in zip(mean.reshape((-1, )), std.reshape((-1, ))):
        labels.append(str(m.round(2)) + '\n' +  r'$\pm$' + str(s.round(2)))
    labels = np.asarray(labels)
    labels = labels.reshape((3, 3))
    
    sns.heatmap(means[0].round(2), annot=labels, fmt = '',
                        xticklabels=classes, yticklabels=classes,
                        cbar = False, ax = ax[1], cmap = cmap)
    ax[1].set_title(r'$n = (2^6, 2 ^ 6, 2^6)$')
    
    classes = [1, 2, 3]
    mean = means[1]
    std = stds[1]
    labels = []
    for m, s in zip(mean.reshape((-1, )), std.reshape((-1, ))):
        labels.append(str(m.round(2)) + '\n' +  r'$\pm$' + str(s.round(2)))
    labels = np.asarray(labels)
    labels = labels.reshape((3, 3))
    sns.heatmap(means[1].round(2), annot=labels, fmt = '',
                        xticklabels=classes, yticklabels=classes,
                        cbar = False, ax = ax[2], cmap = cmap)
    ax[2].set_title(r'$n = (2^4, 2 ^ 6, 2^6)$')
    
    
    classes = [1, 2, 3]
    mean = means[2]
    std = stds[2]
    labels = []
    for m, s in zip(mean.reshape((-1, )), std.reshape((-1, ))):
        labels.append(str(m.round(2)) + '\n' +  r'$\pm$' + str(s.round(2)))
    labels = np.asarray(labels)
    labels = labels.reshape((3, 3))
    sns.heatmap(means[1].round(2), annot=labels, fmt = '',
                        xticklabels=classes, yticklabels=classes,
                        cbar = False, ax = ax[3], cmap = cmap)
    ax[3].set_title(r'$n = (2^2, 2 ^ 6, 2^6)$')
    fig.savefig('plots/SBM-mean-cosine.pdf', bbox_inches = 'tight')
    
    
