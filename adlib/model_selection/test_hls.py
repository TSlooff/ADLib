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
import hls4ml

logging.config.fileConfig('adlib/logging/logging.conf')
logger = logging.getLogger('main')

# parse command line arguments
parser = argparse.ArgumentParser(description='Test script for HLS')
args = parser.parse_args()

data_dir = "./data/"
model_dir = "./model/"

# NOTE: assumes one model file in model folder
model_loc = glob.glob(model_dir + "*")[0]
model = ADModel.load(model_loc)

print(model)
print("window:", len(model.parameters['ae_process_columns']),"x", model.parameters['ae_window'])
hlsmodel = hls4ml.converters.convert_from_pytorch_model(model=model.ae, input_shape=(None, model.parameters[f'ae_l0_nodes']), backend='vitis')
hlsmodel.build()
# print(hlsmodel)