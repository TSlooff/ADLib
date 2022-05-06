#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import csv
import argparse
import pathlib

# parse command line arguments
parser = argparse.ArgumentParser(description='Anomaly Detection algorithm applied to FCD data in given file.')
parser.add_argument('file',
    help='path to the csv file',
    type=pathlib.Path,
    )
parser.add_argument('-c', '--cutoff',
    help='cutoff value for anomaly detection. anomaly scores are in [0,1]. You can specify higher cutoff values, in which case no anomalies will be detected.',
    default=1.0,
    type=float,
    )

args = parser.parse_args()

# HTM imports
from htm.bindings.sdr import SDR
from htm.algorithms import TemporalMemory as TM
from htm.algorithms import SpatialPooler as SP
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.bindings.algorithms import ANMode

# custom encoders
from CustomEncoders import MultiEncoder

# speed encoder
rdse_params_speed = RDSE_Parameters()
rdse_params_speed.size = 1000
rdse_params_speed.resolution = 1
rdse_params_speed.sparsity = 0.02
rdse_speed = RDSE(rdse_params_speed)

# lon and lat encoders
rdse_params_gps = RDSE_Parameters()
rdse_params_gps.size = 1000
rdse_params_gps.resolution = 1e-5
rdse_params_gps.sparsity = 0.02
rdse_lon = RDSE(rdse_params_gps)
rdse_lat = RDSE(rdse_params_gps)

encoder = MultiEncoder([rdse_lat, rdse_lon, rdse_speed])

# set up HTM
sp_params = {
    "inputDimensions":[encoder.get_output_size(),],
    "columnDimensions":[600,],
    "potentialRadius":60,        # default = 16
    "potentialPct":0.5,                         # default = 0.5
    "globalInhibition":True,                    # default = False
    "localAreaDensity":0.02,                    # default = 0.019999999552965164
#     "numActiveColumnsPerInhArea":0,           # default = 0
    "stimulusThreshold":1,                      # default = 0
    "synPermInactiveDec":0.05,                  # default = 0.01
    "synPermActiveInc":0.1,                     # default = 0.1
    "synPermConnected":0.3,                     # default = 0.1
    "minPctOverlapDutyCycle":0.05,              # default = 0.001
    "dutyCyclePeriod":1000,                     # default = 1000
    "boostStrength":1.0,                        # default = 0.0
    "seed":1,                                   # default = 1
    "spVerbosity":0,                            # default = 0
    "wrapAround":False,                         # default = True
}

sp = SP(
    **sp_params
)

# temporal memory
tm_params = {
    "columnDimensions": sp.getColumnDimensions(),
    "cellsPerColumn": 10,                           # default = 32
    "activationThreshold": 1,                      # default = 13
    "initialPermanence": 0.3,                      # default = 0.21
    "connectedPermanence": 0.3,                     # default = 0.5
    "minThreshold": 1,                             # default = 10
    "maxNewSynapseCount": 15,                       # default = 20
    "permanenceIncrement": 0.01,                     # default = 0.1
    "permanenceDecrement": 0.005,                    # default = 0.1
    "predictedSegmentDecrement": 0.002,             # default = 0.0
    "seed": 42,                                     # default = 42
    "maxSegmentsPerCell": 60,                      # default = 255
    "maxSynapsesPerSegment": 40,                   # default = 255
    "checkInputs": True,                            # default = True
    "externalPredictiveInputs": 0,                  # defualt = 0
    "anomalyMode": ANMode.RAW                       # default = ANMode.RAW
}

tm = TM(
    **tm_params
)

with open(args.file, 'r') as f:
    df = np.array(list(csv.reader(f, delimiter=';')))

traj_data = df[:,2:].astype(np.float)

# split data for each vehicle
for vehicle_id in np.unique(df[:, 1]):
    # Slice on the rows with the correct vehicle_id
    # latitude, longitude, speed
    traj = traj_data[np.where(df[:,1] == vehicle_id)]
    
    for i in range(len(traj)):
        sdr_encoding = encoder.encode(traj[i])
        columns = SDR(sp.getColumnDimensions())
        sp.compute(sdr_encoding, True, columns)
        tm.compute(columns, True)

        # trigger based on variable CUTOFF. anomaly score in [0, 1] but is sensitive in early stage so will reach 1.
        if tm.anomaly >= args.cutoff:
            #print("anomaly!")
            sys.exit(1)

    # reset temporal memory for next sequence
    tm.reset()

#print("data parsed. no anomalies.")
sys.exit(0)