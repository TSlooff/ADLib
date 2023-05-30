# AD Lib
General scripts for anomaly detection. Included in this git repository is both the Dockerfile used to create the image, and the Docker-Compose file which uses the image to run the model selection and anomaly detection scripts. 

# Use AD Lib as a docker

## Model Selection
To run the model selection you can use the following command:
`docker-compose run anomaly_detection python adlib/model_selection/run.py --swarming 10 -n 2 --global_time 2`
In this example the swarming optimization is used with 10 particles, 2 processors will be used, and global time indicates the amount of minutes to look for the best model before saving the best found thus far. For all options use the -h or --help flag. 
- The script will look at the first data file in the /app/data folder in the docker, which has a volume to ./data by default. You can change this in the docker-compose file if you want the volume to point somewhere else. 
- The script will output the best model it found to the /app/model folder in the docker, which has a volume to ./model. You can change this as well in the docker-compose file.

## Anomaly Detection
To run the anomaly detection you can use the following command:
`docker-compose run anomaly_detection python adlib/detection/run.py`
- The script will use the first model it finds in /app/model, which has the volume set to ./model/, to perform the detection. It is assumed the data looks like the data used for model selection.
- This model will be used to perform detection on all data files in /app/data, which has the volume set to ./data/, and output a dictionary named `anomalies.json` where the keys are the names of the datafiles, and the values will be the list of indexes where anomalies were found.

# Use AD Lib directly with Python
To use the scripts directly is very similar to using the docker. The only requirement is to set up python (version 3.9 is used in the docker) with the correct dependencies beforehand.
I would recommend to use a virtual environment for this. To set up the requirements:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
git clone https://github.com/htm-community/htm.core
cd ./htm.core/
git checkout 34a6853b7889ecf4174859d3fcd848c45c1c953e
python setup.py install --force
```

In short you can install the dependencies from the requirements.txt file included in this git. Except for htm.core which needs to be installed from source. After installation the htm.core directory can be removed if desired. 

## Model Selection

To run the model selection similarly to the example of the docker, run `python adlib/model_selection/run.py --swarming 10 -n 2 --global_time 2`. 

In this example the swarming optimization is used with 10 particles, 2 processors will be used, and global time indicates the amount of minutes to look for the best model before saving the best found thus far. For all options use the -h or --help flag. 
- The script will look at the first data file in the ./data folder in the docker and use this to try and find the best hyperparameters.
- The script will output the best model it found to the ./model folder. 

## Anomaly Detection
To run the anomaly detection you can use the following command:
`python adlib/detection/run.py`
- The script will use the first model it finds in ./model to perform the detection. It is assumed the data looks like the data used for model selection. No additional metadata file is needed in this stage, as this is stored in the model and it is assumed that the metadata is identical during both stages.
- This model will be used to perform detection on all data files in ./data and output a dictionary named `anomalies.json` where the keys are the names of the datafiles, and the values will be the list of indexes where anomalies were found.

## Config file
Depending on the input data and desired functionality, some additional information may be necessary. This metadata is expected in a json file with an identical name to the input data file. This metadata is only needed once during the model selection stage, as the metadata will subsequently be saved in the model and automatically retrieved for the detection. For numpy arrays saved using numpy.save, and generally csv or excel files, no config should be required unless specific functionality is needed.

Please find below a description of the expected metadata:
|     General               |                                                                                                                                                                                                                                                                                                                                                                      |                                                     |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
|     columns_to_process    |     Columns   to process as list of indices, e.g. [0, 2, 3] (if not given all columns will   be processed)                                                                                                                                                                                                                                                           |                                                     |
|     TODO: resolution            |     Resolution   per column: the smallest significant difference for a column. E.g. for a   celcius temperature 0.5. Expected as list, currently not supported yet.                                                                                                                                                                                                  |
|     TODO: groupby            |     In case of grouped data, this indicates which feature the data is grouped by. E.g. if you have multiagent data, this will indicate which feature identifies the agent.                                                                                                                                                                                                  |                                                      |
|     dtype                 |     Data type   per column. As a list of numpy datatype strings. For valid strings see numpy.sctypeDict                 NOTE: if you use numpy.tofile(),   all shape information is also lost so this should also be included in the   datatype description if it is important. I would recommend to use   numpy.save() instead, as this works much better.          |                                                     |
|     For csv /   excel     |                                                                                                                                                                                                                                                                                                                                                                      |                                                     |
|     csv_delimiter         |     e.g. “,”   (default), “;”, …                                                                                                                                                                                                                                                                                                                                     |                                                     |
|     header                |     None if   the csv does not include a header, otherwise row index of header                                                                                                                                                                                                                                                                                       |                                                     |
|     for   netcdf          |                                                                                                                                                                                                                                                                                                                                                                      |                                                     |
|     netcdf_variable       |    variable   to take for the anomaly detection    |