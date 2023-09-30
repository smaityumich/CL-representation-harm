# packages and path
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from utils import full_allocation_harm_effects, representation_harm, start_cache_list

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}  \usepackage{amssymb}  \usepackage{mathrsfs}'
plt.rc('text', usetex=True)
# Adjust font specs as desired (here: closest similarity to seaborn standard)
plt.rc('font', **{'size': 13.0})
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')


pwd = 'path/to/dir'
reorder = [0, 8, 1, 9, 2, 3, 4, 5, 6, 7]
classes = ['airplane', 'automobile', 
           'bird      ', 'cat',
           'deer', 'dog',
           'frog', 'horse',
           'ship', 'truck']
classes = np.array(classes)
classes_reorder = [classes[i] for i in reorder]

underrepresentation = 100

# # load seeds
# with open(os.path.join(pwd, 'seeds.npy'), 'rb') as fp:
#     seeds = np.load(fp)
# del fp


plot_classes = [0, 8, 1, 9, 2, 4, 7, 3, 5, 6]
# allocation harm plot

file_format = 'features/simsiam_resnet18'

# start_cache_list()

full_allocation_harm_effects(file_format, epochs=500, model_seeds=[0, 1], test_size=0.25,
                             test_seeds=[0, 1], plot_classes=plot_classes,
                             figsize=(16, 7), filename='simsiam_AH_undersampling_100_full.pdf')

representation_harm(file_format, epochs=500, model_seeds=[0, 1], plot_classes=plot_classes,
                             figsize=(7, 7), filename='simsiam_RH_undersampling_100_full.pdf')


        
        
        
# fontsize = 20  
# fontsize2 = 12   
# scale = 0.1   
# fig, ax = plt.subplots()
# plt.axis('off')
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0, 0, '$T$', fontsize = fontsize, verticalalignment='top', bbox=props)    

# ax.text(2 * scale, 0, 'allocation\nharm', fontsize = fontsize,
#         horizontalalignment='center',
#         verticalalignment='center',
#         bbox=props, )        
# ax.text(1 * scale, 1.2 * scale, '$M$', fontsize = fontsize,
#         horizontalalignment='center',
#         verticalalignment='center', bbox=props)   
# ax.set_xlim(-0.1 * scale, 2.4 * scale)    
# ax.set_ylim(-0.4 * scale, 1.4 * scale)    
# ax.annotate("",
#                 xy=(1.7 * scale, -0.1 * scale), xycoords='data',
#                 xytext=(0.12 * scale, -0.11 * scale), textcoords='data',
#                 arrowprops=dict(arrowstyle="->", color="0.5",
#                                 shrinkA=5, shrinkB=5,
#                                 patchA=None, patchB=None,
#                                 connectionstyle="arc3,rad=0.3", lw=3.5,
#                                 ),
#                 )
# ax.text(0.9 * scale, -0.2 * scale, r'$-$rNDE', fontsize = fontsize,
#         horizontalalignment='center',
#         verticalalignment='center',)


# ax.annotate("",
#                 xy=(0.91 * scale, 1.2 * scale), xycoords='data',
#                 xytext=(0.08 * scale, 0.05 * scale), textcoords='data',
#                 arrowprops=dict(arrowstyle="->", color="0.5",
#                                 shrinkA=5, shrinkB=5,
#                                 patchA=None, patchB=None,
#                                 connectionstyle="arc3,rad=-0.3", lw=3.5,
#                                 ),
#                 )

# ax.annotate("",
#                 xy=(1.9 * scale, 0.17 * scale), xycoords='data',
#                 xytext=(1.12 * scale, 1.2 * scale), textcoords='data',
#                 arrowprops=dict(arrowstyle="->", color="0.5",
#                                 shrinkA=5, shrinkB=5,
#                                 patchA=None, patchB=None,
#                                 connectionstyle="arc3,rad=-0.3", lw=3.5,
#                                 ),
#                 )

# ax.text(1.85 * scale, 0.8 * scale, r'NIE', fontsize = fontsize,
#         horizontalalignment='center',
#         verticalalignment='center',)

# plt.savefig(os.path.join(pwd,'plots/mediation.pdf'), bbox_inches = 'tight')

