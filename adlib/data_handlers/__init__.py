#!/usr/bin/env python
# coding: utf-8

from mimetypes import suffix_map
import pathlib
import numpy as np
import logging
import sys
import json
from pandas import DataFrame

logger = logging.getLogger('data_handler')

def get_metadata(file_path: pathlib.Path, verbosity):
    """
    returns the metadata dict given the file path
    """
    meta_loc = file_path.with_suffix('.json')
    if not meta_loc.exists():
        if verbosity > 0:
            logger.warning(f"no metadata: {meta_loc} does not exist")
        return dict()
    with open(meta_loc, "r") as f:
        metadata = json.load(f)
    return metadata

def parse_unknown(file_path, metadata):
    logger.error(f"file extension unknown: {file_path.suffix}")
    sys.exit(1)

def np_to_list_dict(data):
    data = [dict(enumerate(datapoint)) for datapoint in data]
    return data

def slice_cols(data, metadata):
    col_to_process = metadata.get("columns_to_process")
    if col_to_process is None:
        if type(data) == DataFrame:
            col_to_process = list(range(len(data.iloc[0])))
        else:
            if len(data.shape) == 1:
                # only 1 dimension, so add 1
                data = data[:, None]
            col_to_process = list(range(len(data[0])))
        metadata['columns_to_process'] = col_to_process
        return data, metadata
    if type(data) == DataFrame:
        # usecols is used now to slice the dataframes, so nothing to do.
        return data, metadata
    else:
        if len(data.shape) == 1:
            # only 1 dimension, so add 1
            data = data[:, None]
        data = data[:, col_to_process]
    return data, metadata

def parse_numpy_fromfile(file_path, metadata) -> np.array:
    """
    parses given file as numpy array saved with np.tofile().

    NOTE: np.tofile() is suboptimal in that it does not store metadata, so this will need to be put in a metadata file manually by the data owner.
    REQUIRED: "dtype" datatype of the file
    """
    if "dtype" not in metadata.keys():
        logger.error("dtype not in metadata, but is required for this data format.")
        sys.exit(1)
    data = np.fromfile(file_path,  dtype=", ".join(metadata["dtype"]))
    if data.shape[0] == 1: # this can happen when the shape information is given
        data = data[0]
    data, metadata = slice_cols(data, metadata)
    return np_to_list_dict(data), metadata

def parse_numpy_load(file_path, metadata) -> np.array:
    """
    parses given file as if saved with np.save
    necessary metadata for data loading is automatically included with numpy so np.load is enough
    """
    data, metadata = slice_cols(np.load(file_path), metadata)
    return np_to_list_dict(data), metadata

def parse_csv(file_path, metadata) -> np.array:
    """
    parses given csv file
    """
    from pandas import read_csv

    header = metadata.get("header", "infer")
    if header and header.lower() == "none":
        header = None
    mdtype = metadata.get("dtype")
    dtype = {}
    datetime_cols = []
    if mdtype is not None:
        for i in range(len(mdtype)):
            if np.dtype(mdtype[i]) == np.datetime64:
                datetime_cols.append(i)
                dtype[i] = 'object'
            else:
                dtype[i] = mdtype[i]

    data = read_csv(file_path, 
                    delimiter=metadata.get('csv_delimiter'),
                    header=header,
                    usecols=metadata.get("columns_to_process"),
                    dtype=dtype,
                    parse_dates=datetime_cols)
    
    # ensure the columns are always just integers
    data.columns = range(len(data.columns))
    
    data, metadata = slice_cols(data, metadata)
    return data.to_dict("records"), metadata

def parse_netcdf(file_path, metadata) -> np.array:
    """
    parses given netcdf file
    """
    import netCDF4 as nc
    if "netcdf_variable" not in metadata.keys():
        logger.error("netcdf_variable not in metadata, but is required for this data format.")
        sys.exit(1)
    data = nc.Dataset(file_path)[metadata["netcdf_variable"]][:].filled()
    data, metadata = slice_cols(data, metadata)
    return np_to_list_dict(data), metadata

def parse_excel(file_path, metadata) -> np.array:
    """
    parses given excel file
    """
    from pandas import read_excel

    header = metadata.get("header", 0)
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

    data = read_excel(file_path, 
                    header=header,
                    dtype=dtype,
                    usecols=metadata.get("columns_to_process"),
                    parse_dates=datetime_cols)

    # ensure the columns are always just integers
    data.columns = range(len(data.columns))

    data, metadata = slice_cols(data, metadata)

    return data.to_dict("records"), metadata

suffix_map = {
    ".bin": parse_numpy_fromfile,   # numpy array saved with np.tofile
    ".npy": parse_numpy_load,       # numpy array saved with np.save
    ".csv": parse_csv,
    ".nc": parse_netcdf,
    ".xls": parse_excel,
    ".xlsx": parse_excel,
    ".xlsm": parse_excel,
    ".xlsb": parse_excel,
    ".odf": parse_excel,
    ".ods": parse_excel,
    ".odt": parse_excel,
}

def unbyte(metadata: dict):
    # model saving will encode strings to bytes, this is necessary to decode them back to strings.
    for k, v in metadata.items():
        if isinstance(v, bytes):
            metadata[k] = v.decode()
        elif isinstance(v, (list, np.ndarray)):
            for i, j in enumerate(v):
                if isinstance(j, bytes):
                    v[i] = j.decode()
            metadata[k] = v

def parse(file_path: pathlib.Path, metadata = None, verbosity = 0):
    if not file_path.exists():
        logger.error(f"{file_path} does not exist")
        sys.exit(1)
    if metadata:
        unbyte(metadata)
    else:
        # nothing passed
        metadata = get_metadata(file_path, verbosity)
    if verbosity > 0:
        logger.info(f"parsing file: {file_path}")
    # get the function to parse the given suffix, default to unknown if suffix is not in map
    parsing_fn = suffix_map.get(file_path.suffix, parse_unknown)
    return parsing_fn(file_path, metadata)
