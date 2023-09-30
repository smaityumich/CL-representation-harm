# packages and path
import os
import numpy as np
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


pwd = 'path/to/directory'
reorder = [0, 8, 1, 9, 2, 3, 4, 5, 6, 7]
classes = ['airplane', 'automobile', 
           'bird      ', 'cat',
           'deer', 'dog',
           'frog', 'horse',
           'ship', 'truck']
classes = np.array(classes)
classes_reorder = [classes[i] for i in reorder]

underrepresentation = 100

# load seeds
with open(os.path.join(pwd, 'seeds.npy'), 'rb') as fp:
    seeds = np.load(fp)
del fp


plot_classes = [0, 8, 1, 9, 2, 4, 7, 3, 5, 6]
# plot_classes = [0, 8, 1, 9, 4, 7]
# allocation harm plot

file_format = os.path.join(pwd, 'features/simclr_resnet34')

full_allocation_harm_effects(file_format, epochs=500, model_seeds=seeds[:10], test_size=0.25,
                             test_seeds=[0, 1], plot_classes=plot_classes,
                             figsize=(16, 7), filename='AH_undersampling_100_full.pdf')

representation_harm(file_format, epochs=500, model_seeds=seeds[:10], plot_classes=plot_classes,
                             figsize=(7, 7), filename='RH_undersampling_100_full.pdf')


        
        
        
