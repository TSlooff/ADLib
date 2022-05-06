# everest_ad_script

Script to detect anomalies in a given csv file containing FCD data. 

## Requirements
The script is written in Python 3.7. See requirements.txt for all the required packages. Install these before installing htm.core, because htm.core doesn't correctly specify requirements. I strongly recommend to use python 3.7, otherwise you need to install the HTM Core package manually.
HTM Core needs to be installed from test.pypi.org:

```bash
pip install -r requirements.txt
pip install -i https://test.pypi.org/simple/ htm.core
```

## Usage
```bash
 python htm_fcddump.py {csv location}
 python htm_fcddump.py -h
```
OR: 
change the file permissions (once)
```bash
chmod +x htm_fcddump.py
``` 
and you can run the script as a program:
```bash
./htm_fcddump.py {csv location}
./htm_fcddump.py -h
```
The script will exit with 0 if there are no anomalies detected. 
If anomalies are detected the script exits with 1.

## Expected Data
The script assumes a specific structure of the data, otherwise errors may occur. It is the structure of the data that I got to test on, so I hope it is the same. The structure is csv delimited with ```;``` with the following columns: timestamp, vehicle_id, latitude, longitude, speed. See below the first 3 lines of the example csv file.

```
2021-11-16T07:00:08.716Z;V000292;50.15682;14.53224;27.69
2021-11-16T07:00:07.602Z;V000486;50.12362;14.48845;30.65
2021-11-16T07:00:05.008Z;V000516;50.12582;14.44653;30.11
```
