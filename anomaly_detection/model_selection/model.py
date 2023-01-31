#!/usr/bin/env python
# coding: utf-8

from enum import Enum
import numpy as np
import h5py
import pickle

# imports here because they're only necessary if the ADModel is HTM
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM
from htm.algorithms import SpatialPooler as SP
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.bindings.algorithms import ANMode
from htm.algorithms import Predictor

class ModelType(Enum):
    # for now only HTM implemented.
    HTM = 1
    CNN = 2
    GNN = 3

class ADModel:

    def __init__(self):
        pass

    @staticmethod
    def create_model(parameters: dict):
        if parameters['model_type'] == ModelType.HTM.value:
            return ADModelHTM(parameters)
        else:
            raise ValueError(f"model type not supported: {parameters['model_type']}")

    def save(self, file_path: str):
        raise NotImplementedError()

    @staticmethod
    def load(file_path: str):
        with h5py.File(file_path, 'r') as f:
            # will load model, give error if unsupported
            mdl = ADModel.create_model(dict((k,v[()]) for (k,v) in f['parameters'].items()))
            mdl.load(f)
            return mdl

class ADModelHTM(ADModel):

    def __init__(self, parameters: dict):
        super().__init__()

        num_layers = parameters['num_layers']
        # just hardcoded for now. Could determine this from parameter dict but seems unnecessarily complicated for now.
        max_layers = 3

        ## checks for valid parameter values
        if num_layers < 1:
            raise ValueError("num_layers needs to be at least 1")
        elif num_layers > max_layers:
            raise ValueError(f"num layers needs to be smaller than {max_layers} for these parameters")

        # store this for model saving and loading
        self.parameters = parameters

        # set up encoder
        rdse_params = RDSE_Parameters()
        rdse_params.size = parameters['size']
        rdse_params.resolution = parameters['resolution']
        rdse_params.sparsity = 0.02
        self.encoder = RDSE(rdse_params)

        self.tms = [None] * num_layers
        self.sps = [None] * num_layers
        

        for i in range(num_layers):
            if parameters[f'l{i}_minThreshold'] < 1:
                raise ValueError("min threshold needs to be at least 1")

            self.sps[i] = SP(
                inputDimensions            = (self.encoder.size,) if i == 0 else (self.tms[i-1].getActiveCells().size,), # size not dimensions, because the cells are automatically flattened to avoid the increase in dimensions per layer
                columnDimensions           = parameters[f'l{i}_columnDimensions'],
                potentialRadius            = parameters[f'l{i}_potentialRadius'],
                potentialPct               = parameters[f'l{i}_potentialPct'],
                globalInhibition           = True,
                localAreaDensity           = parameters[f'l{i}_localAreaDensity'],
                stimulusThreshold          = int(round(parameters[f'l{i}_stimulusThreshold'])),
                synPermInactiveDec         = parameters[f'l{i}_synPermInactiveDec'],
                synPermActiveInc           = parameters[f'l{i}_synPermActiveInc'],
                synPermConnected           = parameters[f'l{i}_synPermConnected'],
                minPctOverlapDutyCycle     = parameters[f'l{i}_minPctOverlapDutyCycle'],
                dutyCyclePeriod            = int(round(parameters[f'l{i}_dutyCyclePeriod'])),
                boostStrength              = parameters[f'l{i}_boostStrength'],
                seed                       = 0, # this is important, 0="random" seed which changes on each invocation
                spVerbosity                = 99,
                wrapAround                 = False)

            

            self.tms[i] = TM(
                columnDimensions=self.sps[i].getColumnDimensions(),
                cellsPerColumn=parameters[f'l{i}_cellsPerColumn'],                           # default = 32
                activationThreshold=parameters[f'l{i}_activationThreshold'],                      # default = 13
                initialPermanence=parameters[f'l{i}_initialPermanence'],                      # default = 0.21
                connectedPermanence=parameters[f'l{i}_connectedPermanence'],                     # default = 0.5
                minThreshold=parameters[f'l{i}_minThreshold'],                             # default = 10
                maxNewSynapseCount=parameters[f'l{i}_maxNewSynapseCount'],                       # default = 20
                permanenceIncrement=parameters[f'l{i}_permanenceIncrement'],                     # default = 0.1
                permanenceDecrement=parameters[f'l{i}_permanenceDecrement'],                    # default = 0.1
                predictedSegmentDecrement=parameters[f'l{i}_predictedSegmentDecrement'],             # default = 0.0
                seed=0,                                     # default = 42
                maxSegmentsPerCell=parameters[f'l{i}_maxSegmentsPerCell'],                      # default = 255
                maxSynapsesPerSegment=parameters[f'l{i}_maxSynapsesPerSegment'],                   # default = 255
                checkInputs=True,                            # default = True
                externalPredictiveInputs=0,                  # defualt = 0
                anomalyMode=ANMode.RAW                       # default = ANMode.RAW
            )
        
    def detect(self, data, learn=True):
        """
        given the data(point), will calculate the anomaly score of that data.
        """
        input_ = self.encoder.encode(data)
        for i in range(len(self.tms)):
            output_ = SDR(self.sps[i].getColumnDimensions())
            self.sps[i].compute( input_, learn, output_ )
            self.tms[i].compute(output_, learn)
            
            input_ = self.tms[i].getActiveCells().flatten()

        return self.tms[-1].anomaly

    def save(self, file_path: str):
        with h5py.File(file_path, 'w') as f:
            # create parameters group
            param_grp = f.create_group('parameters')
            param_grp.update(self.parameters)

            # NOTE: this is important because dumps-loads will give a segmentation fault otherwise
            for i in range(len(self.tms)):
                self.tms[i].reset()

            #f['columns'] = self.columns.sparse

            f['tms'] = np.frombuffer(pickle.dumps(self.tms), dtype=np.uint8)
            f['sps'] = np.frombuffer(pickle.dumps(self.sps), dtype=np.uint8)

    def load(self, f: h5py.File):
        #self.columns.sparse = f['columns'][()]
        self.sps = pickle.loads(f['sps'][()].tobytes())
        self.tms = pickle.loads(f['tms'][()].tobytes())
