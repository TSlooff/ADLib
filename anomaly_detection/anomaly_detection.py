#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import argparse
import pathlib
import logging
import logging.config
import data_handlers
import glob
import json

from model_selection.model import ADModel

logging.config.fileConfig('logging/logging.conf')
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

# NOTE: assumes one model file in model folder
model_loc = glob.glob("/app/model/*")[0]
model = ADModel.load(model_loc)

anomaly_indexes = {}

for data_loc in glob.glob("/app/data/*"):
    data = data_handlers.parse(pathlib.Path(data_loc))
    for i in range(len(data)):
        # Training Loop
        anomaly_score = model.detect(data[i], learn=True)
        if anomaly_score >= cutoff:
            anomaly_indexes.setdefault(data_loc, []).append(i)
    logger.warning(f"detected {len(anomaly_indexes[data_loc])} anomalies in {data_loc}")
    # TODO
    # model.reset()

# output anomaly indexes to json
with open("/app/data/anomalies.json", "w") as f:
    f.write(json.dumps(anomaly_indexes))

model.save(model_loc)