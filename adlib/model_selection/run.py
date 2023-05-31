import random
import numpy as np
import pandas as pd
import sys
import argparse
import os
import pathlib

# append directory 3 level up to path, allows for importing from adlib module
p = pathlib.Path(__file__).parents[2]
sys.path.append(str(p))

# import multiprocessing
import threading
import time
import re
import htm.optimization.optimizers as optimizers
from adlib.model_selection.custom_swarming import CustomParticleSwarmOptimization, ParamFreezingRule
from functools import partial
import glob
import json
import datetime
from tqdm import tqdm
import faulthandler
faulthandler.enable()

# HTM imports
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM
from htm.algorithms import SpatialPooler as SP
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.bindings.algorithms import ANMode
from htm.algorithms import Predictor

from adlib.data_handlers import parse
from adlib.model_selection.model import ADModel

import adlib.model_selection.ae as optim

with open("adlib/model_selection/default_params.json", "r") as f:
        default_parameters = json.load(f)

        for k, v in default_parameters.items():
            # change lists to tuples because this is required by ae
            if type(v) == list:
                default_parameters[k] = tuple(v)

# to freeze parameters based on number of layers
def freeze_unused_layers(params: dict):
    freeze_keys = set()
    num_layers = params['num_layers']
    f = re.compile("l(\d)_.*")
    for k in params.keys():
        matches = f.findall(k)
        if len(matches) > 0:
            # the key follows the match pattern
            if int(matches[0]) >= num_layers:
                freeze_keys.add("['%s']" % k)
    return freeze_keys

def enforce_min_max(val, min_, max_):
    # ensures value is in [min_, max_] inclusive
    return max(min(max_, val), min_)

def enforce_min(val, min_):
    return max(val, min_)

def enforce_max(val, max_):
    return min(val, max_)

def nobounds(val):
    # no bounds on the value
    return val


# to enforce the bounds of parameters
def enforce_param_bounds(path, params):
    """
    returns a function to enforce a bound on given path, if there is one.
    takes params as input because bound may be parameter-value dependent
    """
    if re.search("num_layers", path):
        return partial(enforce_min_max, min_=1, max_=3)
    
    if re.search("l(\d)_potentialRadius", path):
        return partial(enforce_min, min_=10)
    
    if re.search("l(\d)_boostStrength", path):
        return partial(enforce_min, min_=0)
    
    if re.search("l(\d)_columnDimensions", path):
        return partial(enforce_min, min_=10)
    
    if re.search("l(\d)_dutyCyclePeriod", path):
        return partial(enforce_min, min_=500)
    
    x = re.search("l(\d)_localAreaDensity", path)
    if x:
        return partial(enforce_min_max, min_=max(0.01, 1.5/np.prod(params[f'l{x.groups()[0]}_columnDimensions'])), max_=0.3)
    
    if re.search("l(\d)_minPctOverlapDutyCycle", path):
        return partial(enforce_min_max, min_=0.1, max_=1.0)
    
    if re.search("l(\d)_potentialPct", path):
        return partial(enforce_min_max, min_=0.1, max_=1.0)
    
    if re.search("l(\d)_stimulusThreshold", path):
        return partial(enforce_min, min_=1)
    
    if re.search("l(\d)_synPermActiveInc", path):
        return partial(enforce_min_max, min_=0.01, max_=0.2)
    
    if re.search("l(\d)_synPermConnected", path):
        return partial(enforce_min_max, min_=0.2, max_=0.95)
    
    if re.search("l(\d)_synPermInactiveDec", path):
        return partial(enforce_min_max, min_=0.01, max_=0.2)
    
    if re.search("l(\d)_cellsPerColumn", path):
        return partial(enforce_min, min_=3)
    
    x = re.search("l(\d)_activationThreshold", path)
    if x:
        return partial(enforce_min, min_=params[f"l{x.groups()[0]}_minThreshold"])
    
    if re.search("l(\d)_initialPermanence", path):
        return partial(enforce_min_max, min_=0.1, max_=0.9)
    
    if re.search("l(\d)_connectedPermanence", path):
        return partial(enforce_min_max, min_=0.1, max_=0.9)
    
    if re.search("l(\d)_minThreshold", path):
        return partial(enforce_min, min_=1)
    
    if re.search("l(\d)_maxNewSynapseCount", path):
        return partial(enforce_min, min_=1)
    
    if re.search("l(\d)_permanenceIncrement", path):
        return partial(enforce_min_max, min_=0.01, max_=0.2)
    
    if re.search("l(\d)_permanenceDecrement", path):
        return partial(enforce_min_max, min_=0.01, max_=0.2)
    
    if re.search("l(\d)_predictedSegmentDecrement", path):
        return partial(enforce_min_max, min_=0.01, max_=0.2)
    
    if re.search("l(\d)_maxSegmentsPerCell", path):
        return partial(enforce_min, min_=5)
    
    if re.search("l(\d)_maxSynapsesPerSegment", path):
        return partial(enforce_min, min_=15)
    
    if re.search("encoder_(\d+)_size", path):
        return partial(enforce_min, min_=1000)
    
    if re.search("encoder_(\d+)_resolution", path):
        return partial(enforce_min, min_=0.01)
    
    return nobounds

data_dir = "./data/"
model_dir = "./model/"

def main(parameters, argv=None, verbose=True):
    # set up model and predictor
    # do it before loading data because parameters need to be validated
    data_paths = [d for d in glob.glob(data_dir + "*") if d[-5:] != ".json"]
    data, metadata = parse(pathlib.Path(data_paths[0]))

    admodel = ADModel.create_model(parameters, metadata)
    #pred = Predictor(steps=[1], alpha=parameters['alpha'])

    # # TODO how to handle missing data in general -> don't try to use an incomplete dataset to find a model
    #data[np.where(data == 1e20)] = 0

    test_cut = int(0.9 * len(data))
    train = data[:test_cut]
    test = data[test_cut:]

    
    # Training Loop
    print("training..")
    for i in tqdm(range(len(train))):
        admodel.detect(train[i], learn=True)
        #pred.learn(i, admodel.tms[-1].getActiveCells(), (train[i] / parameters['pred_resolution']).astype('uint'))

    # Testing Loop
    mse = np.zeros_like(metadata['columns_to_process'], dtype='float64')
    print("testing..")
    for i in tqdm(range(len(test) - 1)):
        admodel.detect(test[i], learn=False)
        prediction = admodel.predict()
        #prediction = np.argmax(pred.infer(admodel.tms[-1].getActiveCells())[1])*parameters['pred_resolution']
        # print(pred.infer(admodel.tms[-1].getActiveCells()))
        mse += admodel.decoder.se(prediction, test[i+1])

    mse = mse/(max(len(test)-1, 1))

    print("mse:", mse)
    mse = np.sum(mse)

    print(f'mse:{mse}')
    return -mse # module will look to maximize the output value, so negate it to find the smallest mse

if __name__ == "__main__":

    # mse = main(default_parameters)
    # exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Passed onto the experiment\'s main function.')
    parser.add_argument('--tag', type=str,
        help='Optional string appended to the name of the AE directory.  Use tags to '
                'keep multiple variants of an experiment alive and working at the same time.')
    parser.add_argument('-n', '--processes',  type=int, default=os.cpu_count(),
        help='Number of experiments to run simultaneously, defaults to the number of CPU cores available.')
    parser.add_argument('--time_limit',  type=float, default=None,
        help='Hours, time limit for each run of the experiment.',)
    parser.add_argument('--global_time', type=float, default=15,
        help='Minutes, time limit for the whole script (i.e. all experiments combined)')
    parser.add_argument('--memory_limit',  type=float, default=None,
        help='Gigabytes, RAM memory limit for each run of the experiment.')
    parser.add_argument('--parse',  action='store_true',
        help='Parse the lab report and write it back to the same file, then exit.')
    parser.add_argument('--rmz', action='store_true',
        help='Remove all experiments which have zero attempts.')
    parser.add_argument('experiment', nargs=argparse.REMAINDER,
        help='Name of experiment module followed by its command line arguments.')
    parser.add_argument('-s', '--skip', action='store_true',
        help='In case the model selection can be skipped.')


    all_optimizers = [
        optimizers.EvaluateDefaultParameters,
        optimizers.EvaluateAllExperiments,
        optimizers.EvaluateBestExperiment,
        optimizers.EvaluateHashes,
        optimizers.GridSearch,
        optimizers.CombineBest,
        CustomParticleSwarmOptimization,
    ]
    assert( all( issubclass(Z, optimizers.BaseOptimizer) for Z in all_optimizers))
    for method in all_optimizers:
        method.add_arguments(parser)

    args = parser.parse_args()

    if args.skip:
        print("skip flag")
        exit()

    # default if there is no remainder, since there is no default accepted with argparse.REMAINDER
    if args.experiment == []:
        args.experiment = [__file__]

    selected_method = [X for X in all_optimizers if X.use_this_optimizer(args)]

    # need to retrieve metadata to recreate model
    # need data to set up encoder default parameters
    data_path = pathlib.Path([d for d in glob.glob(data_dir + "*") if d[-5:] != ".json"][0])
    data, metadata = parse(data_path)

    if args.tag is None:
        args.tag = data_path.stem
        print(f"automatically tagged the experiment with {args.tag}")

    # load in default parameters base
    with open("adlib/model_selection/base_params.json", "r") as f:
        default_parameters = json.load(f)

        for k, v in default_parameters.items():
            # change lists to tuples because this is required by ae
            if type(v) == list:
                default_parameters[k] = tuple(v)

    col_to_process = metadata.get('columns_to_process')
    # encoder for each column to process
    default_parameters['num_encoders'] = len(col_to_process)

    encoder_types = ''

    for c, val in enumerate(data[0].values()):
        if isinstance(val, (np.floating, float)):
            # RDSE encoder
            default_parameters[f'encoder_{c}_type'] = 1
            default_parameters[f'encoder_{c}_size'] = 2000
            default_parameters[f'encoder_{c}_resolution'] = 0.05
        elif isinstance(val, (np.datetime64, pd.Timestamp, datetime.datetime)):
            # datetime encoder
            default_parameters[f'encoder_{c}_type'] = 2
        elif isinstance(val, (int, np.integer)):
            # integer, treated as categories
            default_parameters[f'encoder_{c}_type'] = 3
            default_parameters[f'encoder_{c}_size'] = 2000
        elif isinstance(val, (str, np.str)):
            # string, treated as categories but needs to be transformed to integers first.
            default_parameters[f'encoder_{c}_type'] = 4
            default_parameters[f'encoder_{c}_size'] = 2000
        else:
            raise NotImplementedError(f"unsupported data type in data: {type(val)} in column {c}")

        # always add the encoder type to this string since its value should never be changed
        encoder_types += f",['encoder_{c}_type']"

    with open("adlib/model_selection/default_params.json", "w") as f:
        json.dump(default_parameters, f, indent=4)

    ae = optim.Laboratory(experiment_argv=args.experiment,
                          tag      = args.tag,
                          verbose  = args.verbose)
    
    ae.save()
    print("Lab Report written to %s"%ae.lab_report)

    if args.parse:
        pass

    elif args.rmz:
        for x in ae.experiments:
            if x.attempts == 0:
                ae.experiments.remove(x)
                ae.experiment_ids.pop(hash(x))
        ae.save()
        print("Removed all experiments which had not yet been attempted.")

    elif not selected_method:
        print("Error: missing argument for what to to.")
    elif len(selected_method) > 1:
        print("Error: too many argument for what to to.")
    else:
        if selected_method[0] == CustomParticleSwarmOptimization:
            freezing_rules = [
                ParamFreezingRule("['model_type'],['num_encoders']"+encoder_types),
                ParamFreezingRule(freeze_unused_layers),
            ]
            ae.method = selected_method[0](ae, freezing_rules, enforce_param_bounds, args)
        else:
            ae.method = selected_method[0]( ae, args )

        giga = 2**30
        if args.memory_limit is not None:
            memory_limit = int(args.memory_limit * giga)
        else:
            from psutil import virtual_memory
            available_memory = virtual_memory().available
            memory_limit = int(available_memory / args.processes)
            print("Memory Limit %.2g GB per instance."%(memory_limit / giga))


        t = threading.Thread(target = ae.run, args=(args.processes, args.time_limit, memory_limit), daemon=True)
        t.start()
        print(f"running experiments in process for {args.global_time} minutes")
        time.sleep(60*args.global_time)
        ae.finish()
        t.join()

        # ae.run(args.processes, args.time_limit, memory_limit)

        best = max(ae.experiments, key = lambda x: x.mean() )
        # print("best parameters: ", best.parameters)
        mdl_loc = model_dir + "best_model.h5"
        best_mdl = ADModel.create_model(best.parameters, metadata)
        best_mdl.save(mdl_loc)
        print(f"model saved at {mdl_loc}")

        # ae.run( processes    = args.processes,
        #         time_limit   = args.time_limit,
        #         memory_limit = memory_limit,)

    print("Exit.")
    exit(0)
