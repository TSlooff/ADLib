import numpy as np
import pandas as pd
import sys
import argparse
import os
import pathlib

# append directory 3 level up to path, allows for importing from adlib module
p = pathlib.Path(__file__).parents[2]
sys.path.append(str(p))

import logging
import logging.config
import time
import glob
import json
import datetime
from tqdm import tqdm

from adlib.data_handlers import parse
from adlib.model_selection.model import ADModel

import optuna

logging.config.fileConfig('adlib/logging/logging.conf')
logger = logging.getLogger('main')

data_dir = "./data/"
model_dir = "./model/"

def htm_suggestions(params: dict, metadata: dict, trial: optuna.trial.Trial, row):
    params['htm_num_layers'] = trial.suggest_int('htm_num_layers', 1, 3)

    # encoder for each column to process
    params['htm_num_encoders'] = trial.suggest_int('htm_num_encoders', len(metadata.get('columns_to_process')), len(metadata.get('columns_to_process')))
    input_size = 0
    for c, val in enumerate(row):
        if isinstance(val, (np.floating, float)):
            # RDSE encoder
            params[f'htm_encoder_{c}_type'] = 1
            trial.set_user_attr(f'htm_encoder_{c}_type', params[f'htm_encoder_{c}_type'])
            params[f'htm_encoder_{c}_size'] = trial.suggest_int(f'htm_encoder_{c}_size', 500, 10000, step=500)
            params[f'htm_encoder_{c}_resolution'] = trial.suggest_float(f'htm_encoder_{c}_resolution', 0.0001, 100, log=True)
        elif isinstance(val, (np.datetime64, pd.Timestamp, datetime.datetime)):
            # datetime encoder
            params[f'htm_encoder_{c}_type'] = 2
            trial.set_user_attr(f'htm_encoder_{c}_type', params[f'htm_encoder_{c}_type'])
        elif isinstance(val, (int, np.integer)):
            # integer, treated as categories
            params[f'htm_encoder_{c}_type'] = 3
            trial.set_user_attr(f'htm_encoder_{c}_type', params[f'htm_encoder_{c}_type'])
            params[f'htm_encoder_{c}_size'] = trial.suggest_int(f'htm_encoder_{c}_size', 500, 10000, step=500)
        elif isinstance(val, (str, np.str)):
            # string, treated as categories but needs to be transformed to integers first.
            params[f'htm_encoder_{c}_type'] = 4
            trial.set_user_attr(f'htm_encoder_{c}_type', params[f'htm_encoder_{c}_type'])
            params[f'htm_encoder_{c}_size'] = trial.suggest_int(f'htm_encoder_{c}_size', 500, 10000, step=500)
        else:
            raise NotImplementedError(f"unsupported data type in data: {type(val)} in column {c}")
        input_size += params[f'htm_encoder_{c}_size']

    for i in range(params['htm_num_layers']):
        params[f'htm_l{i}_potentialRadius'] = trial.suggest_int(f'htm_l{i}_potentialRadius', 10, int(0.6*input_size))
        params[f'htm_l{i}_boostStrength'] = trial.suggest_float(f'htm_l{i}_boostStrength', 0.0, 3.0, step=0.5)
        params[f'htm_l{i}_columnDimensions'] = trial.suggest_int(f'htm_l{i}_columnDimensions', 100, 10000, step=100)
        params[f"htm_l{i}_dutyCyclePeriod"] = trial.suggest_int(f"htm_l{i}_dutyCyclePeriod", 1000, 100000, step=1000)
        params[f"htm_l{i}_localAreaDensity"] = 0.02
        trial.set_user_attr(f"htm_l{i}_localAreaDensity", params[f"htm_l{i}_localAreaDensity"])
        params[f"htm_l{i}_minPctOverlapDutyCycle"] = trial.suggest_float(f"htm_l{i}_minPctOverlapDutyCycle", 0.01, 0.05, step=0.001)
        params[f"htm_l{i}_potentialPct"] = trial.suggest_float(f"htm_l{i}_potentialPct", 0.2, 0.8, step=0.1)
        params[f"htm_l{i}_stimulusThreshold"] = trial.suggest_int(f"htm_l{i}_stimulusThreshold", 1, 100)
        params[f"htm_l{i}_synPermActiveInc"] = trial.suggest_float(f"htm_l{i}_synPermActiveInc", 0.05, 0.2, step=0.01)
        params[f"htm_l{i}_synPermConnected"] = trial.suggest_float(f"htm_l{i}_synPermConnected", 0.4, 0.95, step=0.05)
        params[f"htm_l{i}_synPermInactiveDec"] = trial.suggest_float(f"htm_l{i}_synPermInactiveDec", 0.01, 0.2, step=0.01)
        params[f"htm_l{i}_cellsPerColumn"] = trial.suggest_int(f"htm_l{i}_cellsPerColumn", 10, 250, step=10)
        params[f"htm_l{i}_minThreshold"] = trial.suggest_int(f"htm_l{i}_minThreshold", 1, 30)
        params[f"htm_l{i}_activationThreshold"] = trial.suggest_int(f"htm_l{i}_activationThreshold", params[f"htm_l{i}_minThreshold"], 50)
        params[f"htm_l{i}_connectedPermanence"] = trial.suggest_float(f"htm_l{i}_connectedPermanence", 0.4, 0.95, step=0.05)
        params[f"htm_l{i}_initialPermanence"] = trial.suggest_float(f"htm_l{i}_initialPermanence", 0.20, round(params[f"htm_l{i}_connectedPermanence"], 2), step=0.05)
        params[f"htm_l{i}_maxSegmentsPerCell"] = trial.suggest_int(f"htm_l{i}_maxSegmentsPerCell", 25, 265, step=20)
        params[f"htm_l{i}_maxSynapsesPerSegment"] = trial.suggest_int(f"htm_l{i}_maxSynapsesPerSegment", 25, 265, step=20)
        params[f"htm_l{i}_maxNewSynapseCount"] = trial.suggest_int(f"htm_l{i}_maxNewSynapseCount", 1, params[f"htm_l{i}_maxSynapsesPerSegment"])
        params[f"htm_l{i}_permanenceIncrement"] = trial.suggest_float(f"htm_l{i}_permanenceIncrement", 0.05, 0.21, step=0.02)
        params[f"htm_l{i}_permanenceDecrement"] = trial.suggest_float(f"htm_l{i}_permanenceDecrement", 0.01, 0.21, step=0.02)
        params[f"htm_l{i}_predictedSegmentDecrement"] = params[f"htm_l{i}_permanenceIncrement"] * params[f"htm_l{i}_localAreaDensity"]
        trial.set_user_attr(f"htm_l{i}_predictedSegmentDecrement", params[f"htm_l{i}_predictedSegmentDecrement"])

def ae_suggestions(params: dict, metadata: dict, trial: optuna.trial.Trial, row):
    params['ae_num_layers'] = trial.suggest_int('ae_num_layers', 1, 10)

    float_columns = []
    for c, val in enumerate(row):
        if isinstance(val, (np.floating, float)):
            # should be processed
            float_columns.append(c)
        else:
            # not processed yet.
            pass
    params['ae_float_columns'] = float_columns
    trial.set_user_attr('ae_float_columns', float_columns)

def suggest(params, metadata, trial, row):
    if params['model_type'] == 1:
        return htm_suggestions(params, metadata, trial, row)
    if params['model_type'] == 2:
        return ae_suggestions(params, metadata, trial, row)
    else:
        raise Exception("Received an incorrect model type")

def main(trial: optuna.trial.Trial):
    params = dict()
    params['model_type'] = trial.suggest_int('model_type', 1, 1)

    data_paths = [d for d in glob.glob(data_dir + "*") if d[-5:] != ".json"]
    data, metadata = parse(pathlib.Path(data_paths[0]))

    suggest(params, metadata, trial, data[0].values())
    admodel = ADModel.create_model(params, metadata)

    test_cut = max(int(0.9 * len(data)),len(data)-500)
    # this is highly imbalanced because the prediction step (which is not necessary for AD)
    # is highly costly so this is reduced as much as possible.
    train = data[:test_cut]
    test = data[test_cut:]

    # Training Loop
    for i in tqdm(range(len(train))):
        admodel.detect(train[i], learn=True)
        #pred.learn(i, admodel.tms[-1].getActiveCells(), (train[i] / parameters['pred_resolution']).astype('uint'))

    # Testing Loop
    se = np.zeros_like(metadata['columns_to_process'], dtype=np.float32)
    for i in tqdm(range(len(test) - 1)):
        admodel.detect(test[i], learn=True)
        prediction = admodel.predict()
        #prediction = np.argmax(pred.infer(admodel.tms[-1].getActiveCells())[1])*parameters['pred_resolution']
        se += admodel.decoder.se(prediction, test[i+1])
        
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
        help='Number of experiments to run simultaneously, defaults to the number of CPU cores available.')
    parser.add_argument('--tag', type=str,
        help='Optional string appended to the name of the AE directory.  Use tags to '
                'keep multiple variants of an experiment alive and working at the same time.')
    parser.add_argument('-s', '--skip', action='store_true',
        help='In case the model selection can be skipped.')
    parser.add_argument('--global_time', type=float, default=2*60, # 2 hours is default
        help='Minutes, time limit for the whole script (i.e. all experiments combined). After timeout current trials will finish before exiting.')
    parser.add_argument('-db', action='store_true',
        help='Indicates the database used by the optimizer should be kept after completion. This database can be reused.')

    args = parser.parse_args()
    if args.skip:
        logger.info("exiting because of skip flag.")
        exit(0)

    data_path = pathlib.Path([d for d in glob.glob(data_dir + "*") if d[-5:] != ".json"][0])
    _, metadata = parse(data_path, metadata=None, verbosity=1)

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
    study.optimize(main, timeout=60*args.global_time, n_jobs=args.processes, gc_after_trial=True)
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