#!/usr/bin/env python
# coding: utf-8

from enum import Enum
import numpy as np
import h5py
import pickle
from collections import defaultdict, Counter

# imports here because they're only necessary if the ADModel is HTM
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM
from htm.algorithms import SpatialPooler as SP
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.encoders.date import DateEncoder
from htm.bindings.algorithms import ANMode

from adlib.model_selection.CustomEncoders import MultiEncoder
import optuna
import torch
from welford import Welford
from scipy.stats import norm

class ModelType(Enum):
    # for now only HTM implemented.
    HTM = 1
    AE = 2
    CAE = 3


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
    params['ae_lr'] = trial.suggest_float('ae_lr', 1e-5, 1e-2, log=True)
    
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

    # input layer
    params['ae_l0_nodes'] = len(float_columns)
    trial.set_user_attr('ae_l0_nodes', params['ae_l0_nodes'])
    params['ae_latent_nodes'] = trial.suggest_int('ae_latent_nodes', 1, int(0.5 * params['ae_l0_nodes']))
    # for simplicity just use one activation throughout
    params['ae_activation'] = trial.suggest_categorical('ae_activation', choices=['relu', 'sigmoid'])

    for l in range(1, params['ae_num_layers']):
        params[f'ae_l{l}_nodes'] = trial.suggest_int(f'ae_l{l}_nodes', params['ae_latent_nodes'], params[f'ae_l{l-1}_nodes'])

    params[f'ae_l{params["ae_num_layers"]}_nodes'] = params['ae_latent_nodes']
    trial.set_user_attr(f'ae_l{params["ae_num_layers"]}_nodes', params[f'ae_l{params["ae_num_layers"]}_nodes'])

def suggest(params, metadata, trial, row):
    params['model_type'] = trial.suggest_int('model_type', 1, 2)
    if params['model_type'] == 1:
        return htm_suggestions(params, metadata, trial, row)
    if params['model_type'] == 2:
        return ae_suggestions(params, metadata, trial, row)
    else:
        raise Exception("Received an incorrect model type")

class ADModel:

    def __init__(self):
        pass

    @staticmethod
    def create_model(parameters: dict, metadata: dict):
        if parameters['model_type'] == ModelType.HTM.value:
            return ADModelHTM(parameters, metadata)
        elif parameters['model_type'] == ModelType.AE.value:
            if type(parameters['ae_activation']) == bytes:
                parameters['ae_activation'] = parameters['ae_activation'].decode()
            return ADModelAE(parameters, metadata)
        else:
            raise ValueError(f"model type not supported: {parameters['model_type']}")

    def save(self, file_path: str):
        raise NotImplementedError()

    @staticmethod
    def load(file_path: str):
        with h5py.File(file_path, 'r') as f:
            # will load model, give error if unsupported
            mdl = ADModel.create_model(dict((k,v[()]) for (k,v) in f['parameters'].items()), dict((k,v[()]) for (k,v) in f['metadata'].items()))
            mdl.load(f)
            return mdl
    
    def SE(self, next_val):
        raise NotImplementedError()

    def reset(self):
        pass

class ADModelAE(ADModel):

    def __init__(self, parameters: dict, metadata: dict):
        super().__init__()
        self.parameters = parameters
        self.metadata = metadata

        self.activation_mapping = {
            'relu': torch.nn.ReLU(),
            'sigmoid': torch.nn.Sigmoid()
        }

        self.float_cols = parameters['ae_float_columns']

        layers = []

        # encoder part
        for l in range(1, parameters['ae_num_layers'] + 1):
            layers.append(torch.nn.Linear(parameters[f'ae_l{l-1}_nodes'], parameters[f'ae_l{l}_nodes']))
            layers.append(self.activation_mapping[parameters['ae_activation']])

        # decoder part
        for l in range(parameters['ae_num_layers'], 1, -1):
            layers.append(torch.nn.Linear(parameters[f'ae_l{l}_nodes'], parameters[f'ae_l{l-1}_nodes']))
            layers.append(self.activation_mapping[parameters['ae_activation']])

        # for output layer the activation function is ommitted to ensure that the model can go to any input value range.
        layers.append(torch.nn.Linear(parameters['ae_l1_nodes'], parameters['ae_l0_nodes']))
        self.ae = torch.nn.Sequential(*layers)

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.ae.parameters(),
                                    lr = parameters['ae_lr'])
        
        # used for the anomaly score
        self.welford = Welford()
        
    def detect(self, data, learn=True):
        """
        given the data(point), will calculate the anomaly score of that data.
        """
        data = torch.Tensor(list(data.values()))
        reconstructed = self.ae.forward(data)
        loss = self.loss_fn(reconstructed, data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.welford.add(np.array(loss.item()))
        self.se = torch.square(reconstructed - data).detach().numpy()
        return norm.cdf(loss.item(), loc=self.welford.mean, scale=np.sqrt(self.welford.var_p))

    def SE(self, next_val):
        return self.se

    def save(self, file_path: str):
        with h5py.File(file_path, 'w') as f:
            param_grp = f.create_group('parameters')
            param_grp.update(self.parameters)
            metadata_grp = f.create_group('metadata')
            metadata_grp.update(self.metadata)
            f['ae_state_dict'] = np.frombuffer(pickle.dumps(self.ae.state_dict()), dtype=np.uint8)
            f['opt_state_dict'] = np.frombuffer(pickle.dumps(self.optimizer.state_dict()), dtype=np.uint8)
            f['welford'] = np.frombuffer(pickle.dumps(self.welford), dtype=np.uint8)
            

    def load(self, f: h5py.File):
        self.ae.load_state_dict(pickle.loads(f['ae_state_dict'][()].tobytes()))
        self.optimizer.load_state_dict(pickle.loads(f['opt_state_dict'][()].tobytes()))
        self.welford = pickle.loads(f['welford'][()].tobytes())


class ADModelHTM(ADModel):

    def __init__(self, parameters: dict, metadata: dict):
        super().__init__()
        
        self.datetime_cols = []
        arithmatic_cols = []
        category_cols = []

        # set up encoders
        num_encoders = parameters['htm_num_encoders']
        encoders = [None] * num_encoders
        for e in range(num_encoders):
            if parameters[f'htm_encoder_{e}_type'] == 1:
                # set up float encoder
                rdse_params = RDSE_Parameters()
                rdse_params.size = parameters[f'htm_encoder_{e}_size']
                rdse_params.resolution = parameters[f'htm_encoder_{e}_resolution']
                rdse_params.sparsity = 0.02
                encoders[e] = RDSE(rdse_params)
                arithmatic_cols.append(e)
            elif parameters[f'htm_encoder_{e}_type'] == 2:
                # set up datetime encoder
                encoders[e] = DateEncoder(weekend=100, timeOfDay=1000, dayOfWeek=900)
                self.datetime_cols.append(e)
            elif parameters[f'htm_encoder_{e}_type'] == 3:
                # set up category encoder for integers.
                rdse_params = RDSE_Parameters()
                rdse_params.size = parameters[f'htm_encoder_{e}_size']
                rdse_params.category = True
                rdse_params.sparsity = 0.02
                encoders[e] = RDSE(rdse_params)
                category_cols.append(e)
            elif parameters[f'htm_encoder_{e}_type'] == 4:
                # string: set up string encoder and category encoder
                rdse_params = RDSE_Parameters()
                rdse_params.size = parameters[f'htm_encoder_{e}_size']
                rdse_params.category = True
                rdse_params.sparsity = 0.02
                encoders[e] = StringEncoder(RDSE(rdse_params))
                category_cols.append(e)

        self.encoder = MultiEncoder(encoders)
        self.decoder = SDRDecoder(arithmatic_cols, self.datetime_cols, category_cols)

        num_layers = parameters['htm_num_layers']
        # just hardcoded for now.
        max_layers = 3

        ## checks for valid parameter values
        if num_layers < 1:
            raise ValueError("htm_num_layers needs to be at least 1")
        elif num_layers > max_layers:
            raise ValueError(f"htm_num_layers needs to be smaller than {max_layers} for these parameters")

        # store this for model saving and loading
        self.parameters = parameters
        self.metadata = metadata

        self.tms = [None] * num_layers
        self.sps = [None] * num_layers
        

        for i in range(num_layers):
            if parameters[f'htm_l{i}_minThreshold'] < 1:
                raise ValueError("htm_min_threshold needs to be at least 1")

            self.sps[i] = SP(
                inputDimensions            = (self.encoder.size,) if i == 0 else (self.tms[i-1].getActiveCells().size,), # size not dimensions, because the cells are automatically flattened to avoid the increase in dimensions per layer
                columnDimensions           = [parameters[f'htm_l{i}_columnDimensions']],
                potentialRadius            = parameters[f'htm_l{i}_potentialRadius'],
                potentialPct               = parameters[f'htm_l{i}_potentialPct'],
                globalInhibition           = True,
                localAreaDensity           = parameters[f'htm_l{i}_localAreaDensity'],
                stimulusThreshold          = parameters[f'htm_l{i}_stimulusThreshold'],
                synPermInactiveDec         = parameters[f'htm_l{i}_synPermInactiveDec'],
                synPermActiveInc           = parameters[f'htm_l{i}_synPermActiveInc'],
                synPermConnected           = parameters[f'htm_l{i}_synPermConnected'],
                minPctOverlapDutyCycle     = parameters[f'htm_l{i}_minPctOverlapDutyCycle'],
                dutyCyclePeriod            = parameters[f'htm_l{i}_dutyCyclePeriod'],
                boostStrength              = parameters[f'htm_l{i}_boostStrength'],
                seed                       = 0, # this is important, 0="random" seed which changes on each invocation
                spVerbosity                = 0,
                wrapAround                 = False)

            self.tms[i] = TM(
                columnDimensions=self.sps[i].getColumnDimensions(),
                cellsPerColumn=parameters[f'htm_l{i}_cellsPerColumn'],                           # default = 32
                activationThreshold=parameters[f'htm_l{i}_activationThreshold'],                      # default = 13
                initialPermanence=parameters[f'htm_l{i}_initialPermanence'],                      # default = 0.21
                connectedPermanence=parameters[f'htm_l{i}_connectedPermanence'],                     # default = 0.5
                minThreshold=parameters[f'htm_l{i}_minThreshold'],                             # default = 10
                maxNewSynapseCount=parameters[f'htm_l{i}_maxNewSynapseCount'],                       # default = 20
                permanenceIncrement=parameters[f'htm_l{i}_permanenceIncrement'],                     # default = 0.1
                permanenceDecrement=parameters[f'htm_l{i}_permanenceDecrement'],                    # default = 0.1
                predictedSegmentDecrement=parameters[f'htm_l{i}_predictedSegmentDecrement'],             # default = 0.0
                seed=0,                                     # default = 42
                maxSegmentsPerCell=parameters[f'htm_l{i}_maxSegmentsPerCell'],                      # default = 255
                maxSynapsesPerSegment=parameters[f'htm_l{i}_maxSynapsesPerSegment'],                   # default = 255
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

        # add to decoder
        self.decoder.map(self.tms[-1].getActiveCells(), data)
        return self.tms[-1].anomaly

    def predict(self, learn=True):
        """
        used to predict the next value. 
        NOTE: Should be called after detect!
        """
        self.tms[-1].activateDendrites(learn)
        return self.decoder.get_data(self.tms[-1].getPredictiveCells())

    def save(self, file_path: str):
        with h5py.File(file_path, 'w') as f:
            # create parameters group
            param_grp = f.create_group('parameters')
            param_grp.update(self.parameters)

            metadata_grp = f.create_group('metadata')
            metadata_grp.update(self.metadata)

            # NOTE: this is important because dumps-loads will give a segmentation fault otherwise
            self.reset()

            # save some stateful attributes using pickle
            f['tms'] = np.frombuffer(pickle.dumps(self.tms), dtype=np.uint8)
            f['sps'] = np.frombuffer(pickle.dumps(self.sps), dtype=np.uint8)
            f['encoder'] = np.frombuffer(pickle.dumps(self.encoder), dtype=np.uint8)
            f['decoder'] = np.frombuffer(pickle.dumps(self.decoder), dtype=np.uint8)

    def load(self, f: h5py.File):
        #self.columns.sparse = f['columns'][()]
        self.sps = pickle.loads(f['sps'][()].tobytes())
        self.tms = pickle.loads(f['tms'][()].tobytes())
        self.encoder = pickle.loads(f['encoder'][()].tobytes())
        self.decoder = pickle.loads(f['decoder'][()].tobytes())

    def reset(self):
        """
        resets the temporal state of the model
        """
        for i in range(len(self.tms)):
            self.tms[i].reset()

    def SE(self, next_val):
        return self.decoder.se(self.predict(), next_val)

class SDRDecoder:
    def __init__(self, arithmatic_cols:list, datetime_cols: list, category_cols: list) -> None:
        # pertinant to columns with floats
        self.arithmatic_cols = arithmatic_cols
        # self.output_size = len(self.arithmatic_cols)
        self.sdr_to_data = dict()
        self.sparse_to_data = defaultdict(Counter)
        
        # pertinant to columns with datetimes.
        self.datetime_cols = datetime_cols
        self.latest_datetimes = dict()

        # pertinant to columns with integers / strings
        self.category_cols = category_cols
        self.sdr_to_categories = dict()
        self.sparse_to_categories = defaultdict(Counter)

    def map(self, sdr: SDR, data):
        """
        maps the given SDR to the given data.
        """
        k = tuple(sdr.sparse)
        #data = np.array(list(data.values()))

        if self.datetime_cols: # if list not empty, extract the datetimes
            self.latest_datetimes = dict()
            for datetime_col in self.datetime_cols:
                self.latest_datetimes[datetime_col] = data[datetime_col]

        if self.category_cols: # if list not empty, extract categories
            old_category_data = self.sdr_to_categories.get(k)
            category_data = tuple([data[c] for c in self.category_cols])
            if old_category_data is not None:
                for c in sdr.sparse:
                    self.sparse_to_categories[c][old_category_data] -= 1
                    if self.sparse_to_categories[c][old_category_data] == 0:
                        del self.sparse_to_categories[c][old_category_data]
                    self.sparse_to_categories[c][category_data] += 1
            else:
                for c in sdr.sparse:
                    self.sparse_to_categories[c][category_data] += 1
            self.sdr_to_categories[k] = category_data

        if self.arithmatic_cols:
            old_arithm_data = self.sdr_to_data.get(k)
            arithm_data = tuple([data[c] for c in self.arithmatic_cols])
            if old_arithm_data is not None:
                # there was a previous mapping
                #old_data = tuple(old_data)
                for c in sdr.sparse:
                    self.sparse_to_data[c][old_arithm_data] -= 1
                    if self.sparse_to_data[c][old_arithm_data] == 0:
                        del self.sparse_to_data[c][old_arithm_data]
                    self.sparse_to_data[c][arithm_data] += 1
            else:
                for c in sdr.sparse:
                    self.sparse_to_data[c][arithm_data] += 1
            self.sdr_to_data[k] = arithm_data        

    def get_data(self, sdr):
        """
        gets the data best corresponding to this sdr
        """
        # note: can not use no_match in the default because it will be evaluated regardless, which is costly.
        k = tuple(sdr.sparse)

        out = dict()
        if self.arithmatic_cols:
            arithm_out = self.sdr_to_data.get(k) # this is a tuple
            if arithm_out is None:
                arithm_out = self.__no_arithm_match(k) # this is a numpy array

            # add to output dict
            out.update((k, v) for k,v in zip(self.arithmatic_cols, arithm_out))

        if self.category_cols:
            category_out = self.sdr_to_categories.get(k) # this is a tuple
            if category_out is None:
                category_out = self.__no_category_match(k)

            # tuple to dict
            out.update((k, v) for k,v in zip(self.category_cols, category_out))
        
        if self.datetime_cols:
            out.update(self.latest_datetimes)
        return out

    def __no_category_match(self, sparse_tuple):
        """
        computes a category value when there is no exact match for the sdr
        """
        # count data values per column
        tracking = [Counter() for _ in self.category_cols]
        
        for c in sparse_tuple:
            for data, d_count in self.sparse_to_categories.get(c, dict()).items():
                for i, v in enumerate(data):
                    tracking[i][v] += d_count
        
        # output mode per column
        return [c.most_common(1)[0][0] if len(c) > 0 else 0 for c in tracking]

    def __no_arithm_match(self, sparse_tuple):
        """
        computes an arithmatic data value when there is no exact match for the sdr
        """
        w_dict = dict()
        
        for c in sparse_tuple:
            for data, d_count in self.sparse_to_data.get(c, dict()).items():
                d_tot = w_dict.get(data, 0)
                w_dict[data] = d_tot + d_count

        out = np.zeros_like(self.arithmatic_cols, dtype=float)
        total = 0
        for d, d_count in w_dict.items():
            if d_count >= int(len(sparse_tuple) / 5):
                # at least some percentage of overlap with sdr to be taken into account
                out += d_count * np.array(d)
                total += d_count
        if total > 0:
            out /= total
        return out

    def se(self, prediction, actual_data):
        """
        calculates the squared error between prediction and the actual data
        """
        error = np.zeros(len(actual_data))
        for col in actual_data.keys():
            if col in self.datetime_cols:
                # datetime column so compare datetimes and get difference in seconds
                error[col] = (prediction[col] - actual_data[col]).total_seconds()
            elif col in self.category_cols:
                # categorical
                # if the same: error = 0
                # if different: error = 1             
                error[col] = 1 - int(prediction[col] == actual_data[col])
            else:
                error[col] = prediction[col] - actual_data[col]
        return error**2

class StringEncoder:
    def __init__(self, encoder) -> None:
        self.idx = 1
        self.string_int = dict()
        self.int_string = dict()
        self.encoder = encoder
        self.size = self.encoder.size

    def encode(self, string):
        """
        converts the given string to an SDR using an internal mapping and RDSE encoder.
        if there is no mapping for this string yet, it is added and encoded.
        """
        if string not in self.string_int:
            # no mapping yet. Add mapping
            self.string_int[string] = self.idx
            self.int_string[self.idx] = string
            self.idx += 1
        return self.encoder.encode(self.string_int[string])

    def decode(self, int):
        """
        converts the given string ID to the string.
        if the given integer does not correspond to an existing string, None is returned
        """
        return self.int_string.get(int)