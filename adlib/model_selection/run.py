import numpy as np
import sys
import argparse
import os
import pathlib

# append directory 3 level up to path, allows for importing from adlib module
p = pathlib.Path(__file__).parents[2]
sys.path.append(str(p))

import logging
import logging.config
import glob
from tqdm import tqdm

from adlib.data_handlers import parse
from adlib.model_selection.model import ADModel, suggest
import optuna
from functools import partial

logging.config.fileConfig('adlib/logging/logging.conf')
logger = logging.getLogger('main')

data_dir = "./data/"
model_dir = "./model/"

def main(trial: optuna.trial.Trial, data, metadata, disable):
    params = dict()

    # data_paths = [d for d in glob.glob(data_dir + "*") if d[-5:] != ".json"]
    # data, metadata = parse(pathlib.Path(data_paths[0]))

    suggest(params, metadata, trial, data[0].values())
    admodel = ADModel.create_model(params, metadata)

    test_cut = max(int(0.9 * len(data)),len(data)-500)
    # this is highly imbalanced because the prediction step (which is not necessary for AD)
    # is highly costly so this is reduced as much as possible.
    train = data[:test_cut]
    test = data[test_cut:]

    # Training Loop
    for i in tqdm(range(len(train)), disable=disable):
        admodel.detect(train[i], learn=True)
        #pred.learn(i, admodel.tms[-1].getActiveCells(), (train[i] / parameters['pred_resolution']).astype('uint'))

    # Testing Loop
    se = np.zeros_like(admodel.processed_columns, dtype=np.float32)
    for i in tqdm(range(len(test) - 1), disable=disable):
        admodel.detect(test[i], learn=True)
        se += admodel.SE(test[i+1])
        
        # support early pruning by Optuna
        trial.report(np.sum(se / (i+1)), i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mse = se/(max(len(test)-1, 1))
    mse = np.sum(mse)
    return mse # return mean squared error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--processes',  type=int, default=-1,
        help='Number of trials to run simultaneously, defaults to the number of CPU cores available.')
    parser.add_argument('--tag', type=str,
        help='Name of the experiment, will be used to name database. Defaults to name of input data file.')
    parser.add_argument('-s', '--skip', action='store_true',
        help='This flag skips the optimization and causes immediate exit.')
    parser.add_argument('--global_time', type=float, default=2*60, # 2 hours is default
        help='Time limit for the optimization in minutes. After timeout current trials will finish before exiting.')
    parser.add_argument('-db', action='store_true',
        help='Indicates the database used by the optimizer should be kept after completion. This database can be reused.')
    parser.add_argument('-p', action='store_true',
        help='This flag disables the progress bar for each trial.')

    args = parser.parse_args()
    if args.skip:
        logger.info("exiting because of skip flag.")
        exit(0)

    data_path = pathlib.Path([d for d in glob.glob(data_dir + "*") if d[-5:] != ".json"][0])
    data, metadata = parse(data_path, metadata=None, verbosity=1)

    if args.tag is None:
        args.tag = data_path.stem
        logger.info(f"automatically named study: {args.tag}")

    storage_name = f"sqlite:///{args.tag}.db"
    study = optuna.create_study(study_name=args.tag, direction='minimize', storage=storage_name, load_if_exists=True)

    logger.info(f"timeout after {args.global_time} minutes")
    if args.db:
        logger.info(f"keeping database after completion at {args.tag}.db")
    else:
        logger.info("removing study database after completion")
    study.optimize(partial(main, data=data, metadata=metadata, disable=args.p), timeout=60*args.global_time, n_jobs=args.processes, gc_after_trial=True)
    # logger.info(f"best parameters: {study.best_params}")

    mdl_loc = model_dir + "best_model.h5"
    params = study.best_params
    params.update(study.best_trial.user_attrs) # fixed attributes which are necessary for model creation are stored here. 
    best_mdl = ADModel.create_model(params, metadata)
    best_mdl.save(mdl_loc)
    logger.info(f"model saved at {mdl_loc}")

    if not args.db:
        if os.path.isfile(f"{args.tag}.db"):
            os.remove(f"{args.tag}.db")