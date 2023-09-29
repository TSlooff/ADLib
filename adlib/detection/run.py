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
from collections import deque
from adlib.model_selection.model import ADModel
import os

from tqdm import tqdm

logging.config.fileConfig('adlib/logging/logging.conf')
logger = logging.getLogger('main')

logger.info("starting anomaly detection")

# parse command line arguments
parser = argparse.ArgumentParser(description='Anomaly Detection algorithm applied to all data in data folder.')

parser.add_argument('-c', '--cutoff',
    help='cutoff value for anomaly detection. anomaly scores are in [0,1]. You can specify higher cutoff values, in which case no anomalies will be detected.',
    default=1.0,
    type=float,
    )

args = parser.parse_args()
cutoff = args.cutoff

data_dir = "./data/"
model_dir = "./model/"

# NOTE: assumes one model file in model folder
model_loc = glob.glob(model_dir + "*")[0]
model = ADModel.load(model_loc)

anomaly_indexes = {}

for data_loc in [d for d in glob.glob(data_dir + "*") if d[-5:] != ".json"]: # exclude json files
    data, metadata = parse(pathlib.Path(data_loc), model.metadata)
    for i in tqdm(range(len(data))):
        anomaly_score = model.detect(data[i], learn=True)
        if anomaly_score >= cutoff:
            anoms = anomaly_indexes.get(data_loc)
            if not anoms:
                anoms = deque()
            anoms.append(i)
    anoms = anomaly_indexes.get(data_loc)
    if anoms:
        anomaly_indexes[data_loc] = list(anoms)
    logger.info(f"detected {len(anomaly_indexes.get(data_loc, []))} anomalies in {data_loc}")
    model.reset()

# output anomaly indexes to json
with open(data_dir + "anomalies.json", "w") as f:
    f.write(json.dumps(anomaly_indexes))

model.save(model_loc)