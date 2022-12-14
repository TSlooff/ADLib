#!/usr/bin/env python
# coding: utf-8

from mimetypes import suffix_map
import pathlib
import numpy as np
import logging
import sys

logger = logging.getLogger('data_handler')

def parse_unknown(file_path):
    logger.error(f"file extension unknown: {file_path.suffix}")
    sys.exit(1)

def parse_numpy(file_path) -> np.array:
    """
    parses given file as default numpy file

    NOTE: for now only handles / assumes the way fabien encodes his data.
    """
    return np.fromfile(file_path,  dtype=np.float32)

suffix_map = {
    ".bin": parse_numpy,
}

def parse(file_path: pathlib.Path) -> np.array:
    if not file_path.exists():
        logger.error(f"{file_path} does not exist")
        sys.exit(1)
    logger.info(f"parsing file: {file_path}")
    parsing_fn = suffix_map.get(file_path.suffix, parse_unknown)
    return parsing_fn(file_path)

def to_swarm_csv(data: np.array, csv_path: str) -> None:
    """
    Takes the data and creates a csv from it which can be used for swarming.
    """
    
    pass