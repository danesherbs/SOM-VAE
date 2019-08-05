FROM tensorflow/tensorflow:1.6.0-gpu-py3

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install .

COPY . /SOM-VAE

WORKDIR /SOM-VAE

RUN bash