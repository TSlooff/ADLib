#!/usr/bin/env python
# coding: utf-8

from mimetypes import suffix_map
import pathlib
import numpy as np
import logging
import sys
import json

logger = logging.getLogger('data_handler')

def get_metadata(file_path: pathlib.Path):
    """
    returns the metadata dict given the file path
    """
    meta_loc = file_path.with_suffix('.json')
    if not meta_loc.exists():
        logger.warning(f"no metadata: {meta_loc} does not exist")
        return None
    with open(meta_loc, "r") as f:
        metadata = json.load(f)
    return metadata

def parse_unknown(file_path):
    logger.error(f"file extension unknown: {file_path.suffix}")
    sys.exit(1)

def parse_numpy_fromfile(file_path) -> np.array:
    """
    parses given file as default numpy file

    NOTE: for now only handles / assumes the way fabien encodes his data.
    """
    metadata = get_metadata(file_path)
    if "dtype" not in metadata.keys():
        logger.error("dtype not in metadata, but is required for this data format.")
        sys.exit(1)
    data = np.fromfile(file_path,  dtype=", ".join(metadata["dtype"]))
    if data.shape[0] == 1: # this can happen when the shape information is given
        data = data[0]
    return data, metadata

def parse_numpy_load(file_path) -> np.array:
    """
    parses given file as if saved with np.save
    necessary metadata for data loading is automatically included with numpy so np.load is enough
    """
    return np.load(file_path), get_metadata(file_path)

def parse_csv(file_path) -> np.array:
    """
    parses given csv file
    """
    from pandas import read_csv

    metadata = get_metadata(file_path)
    header = metadata.get("csv_header")
    if header and header.lower() == "none":
        header = None
    mdtype = metadata.get("dtype")
    dtype = {}
    datetime_cols = []
    if mdtype:
        for i in range(len(mdtype)):
            if np.dtype(mdtype[i]) == np.datetime64:
                datetime_cols.append(i)
                dtype[i] = 'object'
            else:
                dtype[i] = mdtype[i]
    data = read_csv(file_path, 
                    delimiter=metadata.get('csv_delimiter'),
                    header=header,
                    dtype=dtype,
                    parse_dates=datetime_cols)
    return data.values, metadata

def parse_netcdf(file_path) -> np.array:
    """
    parses given netcdf file
    """
    import netCDF4 as nc
    metadata = get_metadata(file_path)
    if "netcdf_variable" not in metadata.keys():
        logger.error("netcdf_variable not in metadata, but is required for this data format.")
        sys.exit(1)
    return nc.Dataset(file_path)[metadata["netcdf_variable"]][:].filled(), metadata

suffix_map = {
    ".bin": parse_numpy_fromfile,   # numpy array saved with np.tofile
    ".npy": parse_numpy_load,       # numpy array saved with np.save
    ".csv": parse_csv,
    ".nc": parse_netcdf,
}

def parse(file_path: pathlib.Path) -> np.array:
    if not file_path.exists():
        logger.error(f"{file_path} does not exist")
        sys.exit(1)
    logger.info(f"parsing file: {file_path}")
    # get the function to parse the given suffix, default to unknown if suffix is not in map
    parsing_fn = suffix_map.get(file_path.suffix, parse_unknown)
    return parsing_fn(file_path)
