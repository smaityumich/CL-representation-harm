# Implementation of boosted contrastive learning (BCL)

The codes are based on the repository <https://github.com/MediaBrain-SJTU/BCL>. Please see our `jobs.py` and `train.py` for the hyperparameter values. Steps to reproduce the results: 

1. Run `python jobs.py i` for the values `i = 0-32` to fit the SimCLR models with BCL_I algorithm. 
2. Change `path/to/project` in both `plots.py` and `utils.py` and run `plots.py` to get the final plots. 