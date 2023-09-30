import numpy as np
import torch, os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}  \usepackage{amssymb}  \usepackage{mathrsfs}'
import seaborn as sns
path = "plots"
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)




pis = 2 ** np.arange(-6, 1, dtype = 'float32') / 3
pis = np.flip(pis)
alphas = np.array([0.05, 0.1, 0.2, 0.4])
cosine12 = np.zeros(shape = (pis.shape[0], alphas.shape[0]))
cosine23 = np.zeros(shape = (pis.shape[0], alphas.shape[0]))


for i in range(pis.shape[0]):
    for j in range(alphas.shape[0]):
        
        print('Running for pi: {}, alpha: {} ...'.format(pis[i], alphas[j]))
        K = 3
        Pi = np.ones(shape = (3, ))
        pi = pis[i]
        Pi[0] = pi
        Pi[1:] = (1 - pi)/(Pi[1:].shape[0])
        Pi = np.array(Pi, dtype = 'float32').reshape((-1, 1))
        alpha = alphas[j]
        A = [[0.95, alpha, 0],
             [alpha, 0.95, 0],
             [0, 0, 0.95]]
        A = np.array(A, dtype = 'float32')
        d = 20
        Pi = torch.from_numpy(Pi)
        A = torch.from_numpy(A)
        lr = 0.2
        epochs = 1000
        
        tau = 1.
        scale = 1.
        v_init = torch.normal(0, 1, size = (K, d))
        v_init = scale * v_init / torch.linalg.norm(v_init, dim = 1, keepdim = True) 
        v = Variable(v_init, requires_grad=True)
        for k in range(epochs):
            lr_i = lr * (k + 1) ** (-0.1)
            # normalized representations
            v_normalized = scale * v / torch.linalg.norm(v, dim = 1, keepdim = True)
            
            
            # connected part of the loss
            weights = A @ Pi
            cosines = v_normalized @ v_normalized.T
            loss_conn1 = ((A * Pi) * cosines / tau).sum(dim = 1, keepdim = True) 
            loss_conn = - (loss_conn1 * Pi / weights).sum()
            
            # disconnected part of the loss
            exp_cosines = torch.exp(cosines/tau)
            loss_dis1 = torch.log((exp_cosines * Pi).sum(dim = 1, keepdim = True))
            loss_dis = (loss_dis1 * Pi).sum()
            
            # total loss
            loss = loss_conn + loss_dis
            loss.backward()
            
            # updating the weight matrix after backpropagation
            with torch.no_grad():
                v = v-(lr_i * v.grad.data)
                v = scale * v / torch.linalg.norm(v, dim = 1, keepdim = True)
            v = Variable(v, requires_grad=True)
            
            if k % 200 == 0: 
                print(f'Loss value at iter {k} is {loss.detach().numpy()}')
                
        v = v.detach()
        v_normalized = scale * v / torch.linalg.norm(v, dim = 1, keepdim = True)
        v_normalized = v_normalized.numpy()
        cosines = (v_normalized @ v_normalized.T)
        cosine12[i, j] = cosines[0, 1]
        cosine23[i, j] = cosines[1, 2]
        
fontsize = 20   
# plt.rc('text', usetex=True)
# # Adjust font specs as desired (here: closest similarity to seaborn standard)
plt.rc('font', **{'size': fontsize})
# plt.rc('text.latex', preamble=r'\usepackage{lmodern}')
cmap = 'Spectral'     
cols = ['r', 'b', 'g', 'k']
markers = ['o', '+', 'x', '^']
# fig, axs = plt.subplots(1, 2, figsize = (10, 4))
fig = plt.figure(figsize = (11.5, 3))
ax = fig.add_subplot(131)

alpha = 0.4
A = [[0.95, alpha, 0],
     [alpha, 0.95, 0],
     [0, 0, 0.95]]
A = np.array(A, dtype = 'float32')
labels = A.astype('str')
labels[0,1] = labels[1, 0] = r'$\alpha$'
classes = [1, 2, 3]
# Activating tex in all labels globally

sns.heatmap(A.round(2), annot=labels, fmt = '',
                    xticklabels=classes, yticklabels=classes,
                    cbar = False, ax = ax, cmap = cmap)

ax = fig.add_subplot(132)
for i in range(1, 4):
    alpha = alphas[i]
    col = cols[i]
    marker = markers[i]
    ax.plot(pis, cosine12[:, i], color = col, marker = marker, label = r'$\alpha = {}$'.format(alpha))
    # ax.plot(pis, cosine23[:, i], color = col, marker = 'x', linestyle = '--',)
ax.set_xscale('log', base = 2)
ax.invert_xaxis()
ax.set_xlabel(r'$\pi_1$', fontsize = fontsize)
ax.set_ylabel(r'$\cos(h_1^\star, h_2^\star)$', fontsize = fontsize)
ax.legend(fontsize = fontsize - 5, loc = 'upper left')


ax = fig.add_subplot(133)
for i in range(1, 4):
    alpha = alphas[i]
    col = cols[i]
    marker = markers[i]
    ax.plot(pis, cosine23[:, i], color = col, marker = marker, label = r'$\alpha = {}$'.format(alpha))
    # ax.plot(pis, cosine23[:, i], color = col, marker = 'x', linestyle = '--',)
ax.set_xscale('log', base = 2)
ax.invert_xaxis()
ax.set_xlabel(r'$\pi_1$', fontsize = fontsize)
ax.set_ylabel(r'$\cos(h_2^\star, h_3^\star)$', fontsize = fontsize)
ax.legend(fontsize = fontsize - 5, title_fontsize = fontsize - 5)


plt.subplots_adjust(wspace=0.5)
plt.savefig('cosine_plot.pdf', bbox_inches = 'tight', dpi = 200)
