import os
from itertools import product
import sys

idx_job = int(float(sys.argv[1]))

args = {
        'lr': [0.5, ],
        'epochs': [300, ],
        'weight_decay': [5e-4, ],
        'temperature': [0.2, ],
        'batch_size': [512, ],
        'eval_freq': [10, ],
        'undersample_class': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ],
        'seed': [0, 1, 2]
       }
args_iters = list(product(*list(args.values())))
keys = list(args.keys())
commands = []
for values in args_iters:
    command = 'python train.py BCL_I --bcl --rand_k 1 '
    for key, value in zip(keys, values):
        command += '--{} {} '.format(key, value)
    commands.append(command)


os.system(commands[idx_job])