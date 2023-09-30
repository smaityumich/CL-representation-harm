# packages and path
import os, json, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
file_index = 0

# load seeds
with open(os.path.join(pwd, 'seeds.npy'), 'rb') as fp:
    seeds = np.load(fp)
del fp


def cross_validation(x_train, y_train, random_state = 0):
    grid = {
        'C': [1, 0.1, 0.01, ]
        }
    lr_cv = GridSearchCV(estimator = LogisticRegression(), param_grid = grid, cv=3, n_jobs=-1, verbose = 4)
    lr_cv.fit(x_train, y_train)
    C_best = lr_cv.best_params_['C']
    return LogisticRegression(C= C_best).fit(x_train, y_train)

def confusion_undersampled(file_format, epochs = 500, undersample = 10, model_seed = 6517725, test_size = 0.25, test_seed = 0, train_undersample = False):
    
    # epochs = 500; undersample = 10; model_seed = 6517725; test_size = 0.25; test_seed = 0; train_undersample = False
    
    # load variables
    
   
    
    file = file_format + '_epoch_{}_undersampled_{}_seed_{}_by_{}.npy'.format(epochs, undersample, model_seed,
                                                                              underrepresentation)
    with open(file, 'rb') as f:
        y = np.load(f)
        x = np.load(f)
        _ = np.load(f)
    del f
    x = x / np.linalg.norm(x, axis = 1, keepdims = True)
    
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=test_seed)
    if train_undersample and (undersample != 10):
        index_undersample = np.where(y_train == undersample)[0]
        index_keep = np.random.choice(index_undersample, size = index_undersample.shape[0] // underrepresentation, replace = False)
        x_t = np.concatenate((x_train[index_keep], x_train[y_train != undersample]), axis = 0)
        y_t = np.concatenate((y_train[index_keep], y_train[y_train != undersample]), axis = 0)
        x_train, y_train = x_t, y_t
        del x_t, y_t
    
    # evaluating last layer accuracy
    # lr = LogisticRegression(C = 0.1, class_weight = None).fit(x_train, y_train) # None, or 'balanced'
    lr = cross_validation(x_train, y_train, random_state=test_seed)
    y_pred = lr.predict(x_test)
    confusion = confusion_matrix(y_test, y_pred, normalize='true')
    # confusion = confusion[reorder, :]
    # confusion = confusion[:, reorder]
    # confusion = pd.DataFrame(confusion, columns = classes_reorder, index = classes_reorder)
    
    return confusion#.to_numpy()


def balanced_performance_without_intervention(file_format, epochs, model_seeds = [6517725, ], test_size = 0.25, test_seeds = [6517725, ]):
    
    # performances of balanced SimCLR model + balanced head
    
    model_seeds = np.array(model_seeds, dtype = 'int')
    test_seeds = np.array(test_seeds, dtype = 'int')
    confusion = np.zeros(shape = (model_seeds.shape[0], test_seeds.shape[0], 10, 10))
    
    for i, model_seed in enumerate(model_seeds):
        for j, test_seed in enumerate(test_seeds):
            try:
                confusion[i, j, :, :] = confusion_undersampled(file_format=file_format, epochs = epochs, undersample = 10,
                                                           model_seed=model_seed, test_size=test_size, test_seed=test_seed)
            except:
                confusion[i, j, :, :] = np.nan
            
    return confusion

def balanced_performance_with_intervention(file_format, epochs, undersample = 1, model_seeds = [6517725, ], test_size = 0.25, test_seeds = [6517725, ]):
    
    # performances of imbalanced SimCLR + balanced head
    
    model_seeds = np.array(model_seeds, dtype = 'int')
    test_seeds = np.array(test_seeds, dtype = 'int')
    confusion = np.zeros(shape = (model_seeds.shape[0], test_seeds.shape[0], 10, 10))
    
    for i, model_seed in enumerate(model_seeds):
        for j, test_seed in enumerate(test_seeds):
            try:
                confusion[i, j, :, :] = confusion_undersampled(file_format=file_format, epochs = epochs, 
                                                               undersample = undersample,
                                                               model_seed=model_seed, test_size=test_size, test_seed=test_seed)
            except:
                confusion[i, j, :, :] = np.nan
            
    return confusion


def imbalanced_performance_with_intervention(file_format, epochs, undersample = 1, model_seeds = [6517725, ], test_size = 0.25, test_seeds = [6517725, ]):
    
    # performances of imbalanced SimCLR + imbalanced head
    
    model_seeds = np.array(model_seeds, dtype = 'int')
    test_seeds = np.array(test_seeds, dtype = 'int')
    confusion = np.zeros(shape = (model_seeds.shape[0], test_seeds.shape[0], 10, 10))
    
    for i, model_seed in enumerate(model_seeds):
        for j, test_seed in enumerate(test_seeds):
            try:
                confusion[i, j, :, :] = confusion_undersampled(file_format=file_format, epochs = epochs, 
                                                               undersample = undersample, model_seed=model_seed,
                                                               test_size=test_size, test_seed=test_seed, train_undersample=True)
            except:
                confusion[i, j, :, :] = np.nan
            
    return confusion

def part(df, classes):
    df = df[classes]
    df = df.loc[classes]
    return df


def mean_std_labels(mean, std, digits = 3):
    shape = mean.shape
    labels = []
    for m, s in zip(mean.reshape((-1, )), std.reshape((-1, ))):
        if np.isnan(s):
            labels.append(str(m.round(digits)))
        else:
            labels.append(str(m.round(digits)) + '\n' +  r'$\pm$' + str(s.round(digits)))
    labels = np.asarray(labels)
    return labels.reshape(shape)


def reduce_by_row(x):
    temp = np.zeros(shape = x.shape[1:])
    for i in range(x.shape[0]):
        temp[i, :] = x[i, i, :]
    return temp

def reduce_rows_cols(x, plot_classes = [0, 8, 1, 9,]):
    y = np.copy(x)
    y = y[:, plot_classes]
    return y[plot_classes, :]


def effects(file_format, epochs, undersample = 1, model_seeds = [6517725, 4930113, 6174026, 4471645, 8057434, ],
            test_size = 0.25, test_seeds = [6517725, 4930113, 6174026, 4471645, 8057434, ], 
            plot = False, classes_plot = ['airplane', 'ship', 'automobile', 'truck']):
    
    confusion_bal = balanced_performance_without_intervention(file_format, epochs, model_seeds = model_seeds, test_size = test_size, test_seeds = test_seeds)
    confusion_imbal_with_intervene = imbalanced_performance_with_intervention(file_format, epochs, undersample=undersample, model_seeds = model_seeds, test_size = test_size, test_seeds = test_seeds)
    confusion_bal_with_intervene = balanced_performance_with_intervention(file_format, epochs, undersample=undersample, model_seeds = model_seeds, test_size = test_size, test_seeds = test_seeds)
    
    confusion_bal_mean = pd.DataFrame(np.nanmean(confusion_bal, axis = (0, 1)), columns=classes, index = classes)
    confusion_imbal_with_intervene_mean = pd.DataFrame(np.nanmean(confusion_imbal_with_intervene, axis = (0, 1)), columns=classes, index = classes)
    confusion_bal_with_intervene_mean = pd.DataFrame(np.nanmean(confusion_bal_with_intervene, axis = (0, 1)), columns=classes, index = classes)
    
    const = 1/ np.sqrt(1000 * test_size)
    confusion_bal_std = const * pd.DataFrame(np.nanstd(confusion_bal, axis = (0, 1)), columns=classes, index = classes)
    confusion_imbal_with_intervene_std = const * pd.DataFrame(np.nanstd(confusion_imbal_with_intervene, axis = (0, 1)), columns=classes, index = classes)
    confusion_bal_with_intervene_std = const * pd.DataFrame(np.nanstd(confusion_bal_with_intervene, axis = (0, 1)), columns=classes, index = classes)
    
        
    TE_mean = part(confusion_imbal_with_intervene_mean, classes_plot).to_numpy() - part(confusion_bal_mean, classes_plot).to_numpy()
    TE_std = part(confusion_imbal_with_intervene_std, classes_plot).to_numpy() + part(confusion_bal_std, classes_plot).to_numpy()
    TE_labels = mean_std_labels(TE_mean, TE_std)
    
    IE_mean = part(confusion_bal_with_intervene_mean, classes_plot).to_numpy() - part(confusion_bal_mean, classes_plot).to_numpy()
    IE_std = part(confusion_bal_with_intervene_std, classes_plot).to_numpy() + part(confusion_bal_std, classes_plot).to_numpy()
    IE_labels = mean_std_labels(IE_mean, IE_std)
    
    DE_mean = part(confusion_imbal_with_intervene_mean, classes_plot).to_numpy() - part(confusion_bal_with_intervene_mean, classes_plot).to_numpy()
    DE_std = part(confusion_imbal_with_intervene_std, classes_plot).to_numpy() + part(confusion_bal_with_intervene_std, classes_plot).to_numpy()
    DE_labels = mean_std_labels(DE_mean, DE_std)
    
    fig, ax = plt.subplots(1, 3, figsize = (14, 4))
    cmap = 'Spectral'
    sns.heatmap(TE_mean, annot=TE_labels, fmt = '',
                        xticklabels=classes_plot, yticklabels=classes_plot,
                        cbar = False, ax = ax[0], cmap = cmap)
    
    ax[0].set_title('TE')
    
    sns.heatmap(DE_mean, annot=DE_labels, fmt = '',
                        xticklabels=classes_plot, 
                        cbar = False, ax = ax[1], cmap = cmap)
    
    ax[1].set_title('rNDE')
    ax[1].set_yticks([])
    
    sns.heatmap(IE_mean, annot=IE_labels, fmt = '',
                        xticklabels=classes_plot, 
                        cbar = False, ax = ax[2], cmap = cmap)
    
    ax[2].set_title('NIE')
    ax[2].set_yticks([])
    plt.show()
    
    
def cache_AH(file_format, epochs = 500, model_seeds = [6517725, ], test_seeds = [6517725, ], test_size = 0.25):
    job = {'file_format': file_format,
           'epochs': epochs, 
           'model_seeds': model_seeds.astype('int32').tolist(),
           'test_seeds': list(test_seeds),
           'underrepresentation': underrepresentation,
           }
    # job_list = []
    
    
    file = os.path.join(pwd, 'cache/AH_{}.npy'.format(file_index))
    
    if os.path.isfile(file):
        with open(file, 'rb') as f:
            TE_mean = np.load(f)
            TE_std = np.load(f)
            IE_mean = np.load(f)
            IE_std = np.load(f)
            DE_mean = np.load(f)
            DE_std = np.load(f)
    else:
        
    
        const = 1/ np.sqrt(1000 * test_size)
        
        
        print('computing AH for (T = 0, M = 0) ...')
        st = time.time()
        confusion_bal = balanced_performance_without_intervention(file_format, epochs, 
                                                                  model_seeds = model_seeds,
                                                                  test_size = test_size,
                                                                  test_seeds = test_seeds)
        run_time = time.time() - st
        print('runtime {}\n\n'.format(run_time))
        
        
        confusion_bal_mean = np.nanmean(confusion_bal, axis = (0, 1))
        confusion_bal_std = const * np.nanstd(confusion_bal, axis = (0, 1))    
        
        confusion_imbal_with_intervene = np.zeros(shape = confusion_bal.shape[:2] + (10,) + confusion_bal.shape[2:])
        for undersample in range(10):
            print('computing AH for (T = 1, M = 1) with undersampling class {} ...'.format(undersample))
            st = time.time()
            confusion_imbal_with_intervene[:, :, undersample, :, :] = imbalanced_performance_with_intervention(file_format, epochs, 
                                                                                                               undersample=undersample, 
                                                                                                               model_seeds = model_seeds, 
                                                                                                               test_size = test_size, 
                                                                                                               test_seeds = test_seeds)
            run_time = time.time() - st
            print('runtime {}\n\n'.format(run_time))
            
        confusion_imbal_with_intervene_mean = np.nanmean(confusion_imbal_with_intervene, axis = (0, 1))
        confusion_imbal_with_intervene_mean = reduce_by_row(confusion_imbal_with_intervene_mean)
        confusion_imbal_with_intervene_std = np.nanstd(confusion_imbal_with_intervene, axis = (0, 1))
        confusion_imbal_with_intervene_std = const * reduce_by_row(confusion_imbal_with_intervene_std)
        
        
        confusion_bal_with_intervene = np.zeros_like(confusion_imbal_with_intervene)
        for undersample in range(10):
            print('computing AH for (T = 0, M = 1) with undersampling class {} ...'.format(undersample))
            confusion_bal_with_intervene[:, :, undersample, :, :] = balanced_performance_with_intervention(file_format, epochs, 
                                                                                                           undersample=undersample, 
                                                                                                           model_seeds = model_seeds, 
                                                                                                           test_size = test_size, 
                                                                                                           test_seeds = test_seeds)
            run_time = time.time() - st
            print('runtime {}\n\n'.format(run_time))
            
        confusion_bal_with_intervene_mean = np.nanmean(confusion_bal_with_intervene, axis = (0, 1))
        confusion_bal_with_intervene_mean = reduce_by_row(confusion_bal_with_intervene_mean)
        confusion_bal_with_intervene_std = np.nanstd(confusion_bal_with_intervene, axis = (0, 1))
        confusion_bal_with_intervene_std = const * reduce_by_row(confusion_bal_with_intervene_std) 
        
        TE_mean = confusion_imbal_with_intervene_mean - confusion_bal_mean
        TE_std = np.sqrt(confusion_imbal_with_intervene_std ** 2 + confusion_bal_std ** 2)
        IE_mean = confusion_bal_with_intervene_mean - confusion_bal_mean
        IE_std = np.sqrt(confusion_bal_with_intervene_std ** 2 + confusion_bal_std ** 2)
        DE_mean = confusion_imbal_with_intervene_mean - confusion_bal_with_intervene_mean
        DE_std = np.sqrt(confusion_imbal_with_intervene_std ** 2 + confusion_bal_with_intervene_std ** 2)
        
        with open(file, 'wb') as f:
            np.save(f, TE_mean)
            np.save(f, TE_std)
            np.save(f, IE_mean)
            np.save(f, IE_std)
            np.save(f, DE_mean)
            np.save(f, DE_std)
            
    return TE_mean, TE_std, IE_mean, IE_std, DE_mean, DE_std


    
def full_allocation_harm_effects(file_format, epochs = 500, model_seeds = [6517725, ], test_size = 0.25, test_seeds = [6517725, ], plot_classes = [0, 8, 1, 9,], figsize = (14, 4), filename = None):
    
    # for diagnosis
    # epochs = 500; model_seeds = seeds[:2]; test_size = 0.25; test_seeds = range(2); plot_classes = reorder; figsize = (24, 7); filename = 'AH_undersampling_10.pdf'
    
    TE_mean, TE_std, IE_mean, IE_std, DE_mean, DE_std = cache_AH(file_format, epochs = epochs,\
                                                                 model_seeds = model_seeds, test_seeds = test_seeds,
                                                                 test_size = test_size)

    TE_mean = reduce_rows_cols(TE_mean, plot_classes = plot_classes)     
    TE_std = reduce_rows_cols(TE_std, plot_classes = plot_classes)
    TE_labels = mean_std_labels(TE_mean, TE_std)
    
    IE_mean = reduce_rows_cols(IE_mean, plot_classes = plot_classes)     
    IE_std = reduce_rows_cols(IE_std, plot_classes = plot_classes)
    IE_labels = mean_std_labels(IE_mean, IE_std)
    
    DE_mean = reduce_rows_cols(DE_mean, plot_classes = plot_classes)     
    DE_std = reduce_rows_cols(DE_std, plot_classes = plot_classes)
    DE_labels = mean_std_labels(DE_mean, DE_std)
    
    classes_plot = [classes[i] for i in plot_classes]
    fig, ax = plt.subplots(1, 2, figsize = figsize)
    cmap = 'Spectral'
    sns.heatmap(TE_mean, annot=TE_labels, fmt = '',
                        xticklabels=classes_plot, yticklabels=classes_plot,
                        cbar = False, ax = ax[0], cmap = cmap)
    
    ax[0].set_title('TE')
    
    
    
    sns.heatmap(IE_mean, annot=IE_labels, fmt = '',
                        xticklabels=classes_plot, 
                        cbar = False, ax = ax[1], cmap = cmap)
    
    ax[1].set_title('NIE')
    ax[1].set_yticks([])
    
    # sns.heatmap(DE_mean, annot=DE_labels, fmt = '',
    #                     xticklabels=classes_plot, 
    #                     cbar = False, ax = ax[2], cmap = cmap)
    
    # ax[2].set_title('$-$rNDE')
    # ax[2].set_yticks([])
    
    if type(filename) == str:
        plt.savefig(os.path.join(pwd, 'plots/', filename), bbox_inches = 'tight')
    


def cosine_singe_model(file_format, epochs = 500, undersample = 10, model_seed = 6517725,):
    
    # load variables
    file =  file_format + '_epoch_{}_undersampled_{}_seed_{}_by_{}.npy'.format(epochs, undersample, model_seed,
                                                                               underrepresentation)
    with open(file, 'rb') as f:
        y = np.load(f)
        x = np.load(f)
        _ = np.load(f)
    del f
    for _ in range(5):
        x = x / np.linalg.norm(x, axis = 1, keepdims = True)

    # claculate matric
    n_class = np.unique(y).shape[0]
    cosines_mean = np.zeros(shape = (n_class, n_class))   
    cosines_std = np.zeros(shape = (n_class, n_class))   
    for i in range(n_class):
        for j in range(n_class):
            x_i = x[y == i]
            x_j = x[y == j]
            cosines_ij = x_i @ x_j.T
            cosines_mean[i, j] = cosines_ij.mean()
            cosines_std[i, j] = cosines_ij.std() / np.sqrt(np.prod(cosines_ij.shape))
    return cosines_mean, cosines_std
            
def cosine_metrics(file_format, epochs = 500, undersample = 10, model_seeds = [6517725, ]):
    model_seeds = np.array(model_seeds, dtype = 'int')
    cosine_mean = np.zeros(shape = (model_seeds.shape[0], 10, 10))
    cosine_std = np.zeros(shape = (model_seeds.shape[0], 10, 10))
    for i, model_seed in enumerate(model_seeds):
        try:
            cosine_mean[i, :, :], cosine_std[i, :, :] = cosine_singe_model(file_format, epochs = epochs,
                                                                           undersample = undersample, model_seed=model_seed)
        except:
            cosine_mean[i, :, :], cosine_std[i, :, :] = np.nan, np.nan
    return cosine_mean, cosine_std


def start_cache_list():
    job_list = []
    with open(os.path.join(pwd, 'cache/RH_cache_list.json'), 'w') as f:
        json.dump(job_list, f)
        
    with open(os.path.join(pwd, 'cache/AH_cache_list.json'), 'w') as f:
        json.dump(job_list, f)
        
def cache_RH(file_format, epochs = 500, model_seeds = [6517725, ]):
    
    
    file = os.path.join(pwd, 'cache/RH_{}.npy'.format(file_index))
    
    if os.path.isfile(file):
        with open(file, 'rb') as f:
            RH_mean = np.load(f)
            RH_std = np.load(f)
    else:
    
        cosine_bal, cosine_bal_std = cosine_metrics(file_format, epochs = epochs, model_seeds=model_seeds, undersample=10)
        cosine_bal_mean = np.nanmean(cosine_bal, axis = 0)
        cosine_bal_std = np.nanmean(cosine_bal_std, axis = 0)
        
        cosine_imbal, cosine_imbal_std = np.zeros(shape = (10, ) + cosine_bal.shape), np.zeros(shape = (10, ) + cosine_bal.shape)
        for undersample in range(10):
            cosine_imbal[undersample, :, :, :], cosine_imbal_std[undersample, :, :, :] = cosine_metrics(file_format, 
                                                                                                        epochs = epochs,
                                                                                                        model_seeds=model_seeds,
                                                                                                        undersample=undersample)
        cosine_imbal_mean = np.nanmean(cosine_imbal, axis = 1)
        cosine_imbal_mean = reduce_by_row(cosine_imbal_mean)
        # cosine_imbal_std = np.nanmean(cosine_imbal_std, axis = 1)  
        cosine_imbal_std = np.nanstd(cosine_imbal, axis = 1)  
        cosine_imbal_std = reduce_by_row(cosine_imbal_std)
        
        
        
        RH_mean = (1 - cosine_imbal_mean) / (1 - cosine_bal_mean)
        # RH_std = np.sqrt((cosine_imbal_std ** 2) / ((1 - cosine_imbal_mean) ** 2) + (cosine_bal_std ** 2) / ((1 - cosine_bal_mean) ** 2)) * RH_mean
        RH_std = np.sqrt(cosine_imbal_std ** 2 + cosine_bal_std ** 2)
        
        with open(file, 'wb') as f:
            np.save(f, RH_mean)
            np.save(f, RH_std)
    return RH_mean, RH_std


    
def representation_harm(file_format, epochs = 500, model_seeds = [6517725, ], plot_classes = [0, 8, 1, 9,], figsize = (14, 4), filename = None):    
    
    # for diagnosis
    # epochs = 500; model_seeds = seeds[:2]; plot_classes = reorder; figsize = (7, 7); filename = 'RH_undersampling_100.pdf'
    
    RH_mean, RH_std = cache_RH(file_format, epochs = epochs, model_seeds = model_seeds)
        
        
    RH_mean = reduce_rows_cols(RH_mean, plot_classes=plot_classes)
    RH_std = reduce_rows_cols(RH_std, plot_classes=plot_classes)
    RH_labels = mean_std_labels(RH_mean, RH_std)
    
    
    
    
    classes_plot = [classes[i] for i in plot_classes]
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    cmap = 'Spectral'
    sns.heatmap(RH_mean, annot=RH_labels, fmt = '',
                        xticklabels=classes_plot, yticklabels=classes_plot,
                        cbar = False, ax = ax, cmap = cmap)
    
    if type(filename) == str:
        plt.savefig(os.path.join(pwd, 'plots/', filename), bbox_inches = 'tight')