# based on mnist.py and ae from htm.core
import random
import numpy as np
import pandas as pd
import sys
import argparse
import os
import pathlib
import multiprocessing
import time
import re
import htm.optimization.optimizers as optimizers
from custom_swarming import CustomParticleSwarmOptimization, ParamFreezingRule
from functools import partial

# tracing
import faulthandler
faulthandler.enable()

# HTM imports
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM
from htm.algorithms import SpatialPooler as SP
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.bindings.algorithms import ANMode
from htm.algorithms import Predictor

sys.path.append("../") # to enable this import:
from data_handlers import parse
from model import *

import htm.optimization.ae as optim
# custom encoders
#from encoders.CustomEncoders import MultiEncoder

# starting parameters that will be optimized to the dataset
default_parameters = {
    'model_type': ModelType.HTM.value,

    # predictor
    'alpha': 0.1,
    'pred_resolution': 0.1,

    # encoder
    'size': 2000,
    'resolution': 0.5,

    # because the parameter dict needs to be constant size between experiments, there are always the parameters for each layer
    # and a parameter to indicate how many layers are actually used
    'num_layers': 1,

    ## layer 0
    # spatial pooler 0
    'l0_potentialRadius': 800,
    'l0_boostStrength': 3,
    'l0_columnDimensions': (512,),
    'l0_dutyCyclePeriod': 1000,
    'l0_localAreaDensity': 0.02,
    'l0_minPctOverlapDutyCycle': 0.2,
    'l0_potentialPct': 0.1,
    'l0_stimulusThreshold': 6,
    'l0_synPermActiveInc': 0.07,
    'l0_synPermConnected': 0.5,
    'l0_synPermInactiveDec': 0.02,

    # temporal memory 0
    'l0_cellsPerColumn': 6,
    'l0_activationThreshold': 3,
    'l0_initialPermanence': 0.4,
    'l0_connectedPermanence': 0.5,
    'l0_minThreshold': 1,
    'l0_maxNewSynapseCount': 15,
    'l0_permanenceIncrement': 0.1,
    'l0_permanenceDecrement': 0.05,
    'l0_predictedSegmentDecrement': 0.02,
    'l0_maxSegmentsPerCell': 255,
    'l0_maxSynapsesPerSegment': 255,

    ## layer 1
    # spatial pooler 1
    'l1_potentialRadius': 5000,
    'l1_boostStrength': 3,
    'l1_columnDimensions': (256,),
    'l1_dutyCyclePeriod': 1000,
    'l1_localAreaDensity': 0.02,
    'l1_minPctOverlapDutyCycle': 0.2,
    'l1_potentialPct': 0.1,
    'l1_stimulusThreshold': 6,
    'l1_synPermActiveInc': 0.07,
    'l1_synPermConnected': 0.5,
    'l1_synPermInactiveDec': 0.02,

    # temporal memory 1
    'l1_cellsPerColumn': 6,
    'l1_activationThreshold': 3,
    'l1_initialPermanence': 0.4,
    'l1_connectedPermanence': 0.5,
    'l1_minThreshold': 1,
    'l1_maxNewSynapseCount': 15,
    'l1_permanenceIncrement': 0.1,
    'l1_permanenceDecrement': 0.05,
    'l1_predictedSegmentDecrement': 0.02,
    'l1_maxSegmentsPerCell': 255,
    'l1_maxSynapsesPerSegment': 255,

    ## layer 2
    # spatial pooler 2
    'l2_potentialRadius': 300,
    'l2_boostStrength': 3,
    'l2_columnDimensions': (128,),
    'l2_dutyCyclePeriod': 1000,
    'l2_localAreaDensity': 0.02,
    'l2_minPctOverlapDutyCycle': 0.2,
    'l2_potentialPct': 0.1,
    'l2_stimulusThreshold': 6,
    'l2_synPermActiveInc': 0.07,
    'l2_synPermConnected': 0.5,
    'l2_synPermInactiveDec': 0.02,

    # temporal memory 2
    'l2_cellsPerColumn': 6,
    'l2_activationThreshold': 3,
    'l2_initialPermanence': 0.4,
    'l2_connectedPermanence': 0.5,
    'l2_minThreshold': 1,
    'l2_maxNewSynapseCount': 15,
    'l2_permanenceIncrement': 0.1,
    'l2_permanenceDecrement': 0.05,
    'l2_predictedSegmentDecrement': 0.02,
    'l2_maxSegmentsPerCell': 255,
    'l2_maxSynapsesPerSegment': 255,
}

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
    
    if re.search("l(\d)_minThreshold", path):
        return partial(enforce_min, min_=1)
    
    x = re.search("l(\d)_activationThreshold", path)
    if x:
        return partial(enforce_min, min_=params[f"l{x.groups()[0]}_minThreshold"])
    
    if re.search("l(\d)_boostStrength", path):
        return partial(enforce_min, min_=0)
    
    return nobounds


def main(parameters=default_parameters, argv=None, verbose=True):
    # set up model and predictor
    # do it before loading data because parameters need to be validated
    admodel = ADModel.create_model(parameters)
    pred = Predictor(steps=[1], alpha=parameters['alpha'])

    data = parse(pathlib.Path("../../data/Input-Nimes/MES_Nimes_PHI.bin"))

    # TODO how to handle missing data?
    data[np.where(data == 1e20)] = 0

    test_cut = int(0.8 * len(data))
    train = data[:test_cut]
    test = data[test_cut:]

    
    # Training Loop
    for i in range(len(train)):
        admodel.detect(train[i], learn=True)
        pred.learn(i, admodel.tms[-1].getActiveCells(), int(train[i] / parameters['pred_resolution']))

    # Testing Loop
    mse = 0
    for i in range(len(test) - 1):
        admodel.detect(test[i], learn=False)
        prediction = np.argmax(pred.infer(admodel.tms[-1].getActiveCells())[1])*parameters['pred_resolution']
        mse += (prediction - test[i+1])**2

    mse = mse/(len(test)-1)

    print(f'mse:{mse}')
    return -mse # module will look to maximize the output value, so negate it to find the smallest mse

if __name__=="__main__":
    # main()
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

    # default if there is no remainder, since there is no default accepted with argparse.REMAINDER
    if args.experiment == []:
        args.experiment = [__file__]
    
    selected_method = [X for X in all_optimizers if X.use_this_optimizer(args)]

    ae = optim.Laboratory(args.experiment,
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
                ParamFreezingRule("['model_type']"),
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


        t = multiprocessing.Process(target = ae.run, args=(args.processes, args.time_limit, memory_limit))
        t.start()
        print(f"running experiments in process for {args.global_time} minutes")
        time.sleep(60*args.global_time)
        t.terminate()

        best = max(ae.experiments, key = lambda x: x.mean() )
        print("best parameters: ", best.parameters)
        mdl_loc = "best_model.h5"
        best_mdl = ADModel.create_model(best.parameters)
        best_mdl.save(mdl_loc)
        print(f"model saved at {mdl_loc}")

        # ae.run( processes    = args.processes,
        #         time_limit   = args.time_limit,
        #         memory_limit = memory_limit,)

    print("Exit.")
    # TODO discuss with IT4I also if they have some preference for how the data config script will look like
    # TODO based on input data adapt the experiment
