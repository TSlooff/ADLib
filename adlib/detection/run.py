#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
import argparse
import pathlib

# append directory 3 level up to path, allows for importing from adlib module
p = pathlib.Path(__file__).parents[2]
sys.path.append(str(p))

import logging
import logging.config
from adlib.data_handlers import parse
import glob
import json
from adlib.model_selection.model import ADModel

from tqdm import tqdm

logging.config.fileConfig('adlib/logging/logging.conf')
logger = logging.getLogger('main')

# parse command line arguments
parser = argparse.ArgumentParser(description='Anomaly Detection algorithm applied to all data in data folder.')

parser.add_argument('-c', '--cutoff',
    help='cutoff value for anomaly detection. anomaly scores are in [0,1]. You can specify higher cutoff values, in which case no anomalies will be detected.',
    default=1.0,
    type=float,
    )
parser.add_argument('-p', action='store_true',
    help='This flag disables the progress bar when parsing the files.')

args = parser.parse_args()
cutoff = args.cutoff

data_dir = "./data/"
model_dir = "./model/"

# NOTE: assumes one model file in model folder
model_loc = glob.glob(model_dir + "*")[0]
model = ADModel.load(model_loc)
anomaly_per_file = dict()

logger.info(f"starting anomaly detection using cutoff of {cutoff}")

for data_loc in [d for d in glob.glob(data_dir + "*") if d[-5:] != ".json"]: # exclude json files
    data, metadata = parse(pathlib.Path(data_loc), model.metadata, verbosity=1)
    anomaly_indexes = list()
    for i in tqdm(range(len(data)), disable=args.p):
        anomaly_score = model.detect(data[i], learn=True)
        if anomaly_score >= cutoff:
            anomaly_indexes.append(i)
    anomaly_per_file[data_loc] = anomaly_indexes
    logger.info(f"detected {len(anomaly_indexes)} anomalies in {data_loc}")
    model.reset()

# output anomaly indexes to json
with open(data_dir + "anomalies.json", "w") as f:
    f.write(json.dumps(anomaly_per_file))

model.save(model_loc)