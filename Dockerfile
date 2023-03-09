FROM python:3.9

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Have to install from source now because the image has been pulled from the pip repository
RUN git clone https://github.com/htm-community/htm.core
WORKDIR /app/htm.core/
RUN git checkout 34a6853b7889ecf4174859d3fcd848c45c1c953e
RUN python setup.py install --force

WORKDIR /app

#COPY adlib/ ./adlib/
