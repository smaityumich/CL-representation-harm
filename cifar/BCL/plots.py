# packages and path
import numpy as np
from utils import full_allocation_harm_effects, representation_harm, indirect_effects

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}  \usepackage{amssymb}  \usepackage{mathrsfs}'
plt.rc('text', usetex=True)
# Adjust font specs as desired (here: closest similarity to seaborn standard)
plt.rc('font', **{'size': 13.0})
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')


pwd = 'path/to/project'
reorder = [0, 8, 1, 9, 2, 3, 4, 5, 6, 7]
classes = ['airplane', 'automobile', 
           'bird      ', 'cat',
           'deer', 'dog',
           'frog', 'horse',
           'ship', 'truck']
classes = np.array(classes)
classes_reorder = [classes[i] for i in reorder]

underrepresentation = 100
plot_classes = [0, 8, 1, 9, 4, 7,]

file_format = 'features/bcl_resnet18'

# full_allocation_harm_effects(file_format, epochs=300, model_seeds=[0, 1], test_size=0.25,
#                              test_seeds=[0, 1], plot_classes=plot_classes,
#                              figsize=(16, 7), filename='bcl_AH_undersampling_100_full.pdf')


# representation_harm(file_format, epochs=300, model_seeds=[0, 1], plot_classes=plot_classes,
#                              figsize=(7, 7), filename='bcl_RH_undersampling_100_full.pdf')

# full_allocation_harm_effects(file_format, epochs=300, model_seeds=[0, 1], test_size=0.25,
#                              test_seeds=[0, 1], plot_classes=plot_classes,
#                              figsize=(16, 7), filename='bcl_AH_undersampling_100.pdf')

fig_height = 4.5
indirect_effects(file_format, epochs=300, model_seeds=[0, 1], test_size=0.25,
                             test_seeds=[0, 1], plot_classes=plot_classes,
                             figsize=(fig_height, fig_height), filename='bcl_NIE_undersampling_100.pdf')

representation_harm(file_format, epochs=300, model_seeds=[0, 1], plot_classes=plot_classes,
                             figsize=(fig_height, fig_height), filename='bcl_RH_undersampling_100.pdf')



        
        
        
