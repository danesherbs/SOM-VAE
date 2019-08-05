#FROM anibali/pytorch:cuda-10.0
#FROM ufoym/deepo:pytorch-py36-cu100
#FROM tensorflow/tensorflow:1.10.1-gpu-py3
FROM ufoym/deepo:tensorflow-py36-cu100

#Favourite shell and text editor
RUN apt-get update && apt-get install -y \
    nano \
    wget \
    gcc \
    zsh \
    git \
 && rm -rf /var/lib/apt/lists/*

#Oh My Zsh
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

# Install Requirements
RUN pip install --upgrade pip
COPY requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt
RUN pip uninstall -y tensorflow-estimator-2.0-preview
RUN pip uninstall -y tf-nightly-gpu-2.0-preview
RUN pip uninstall -y tb-nightly
RUN pip install tensorflow-gpu==1.13.1

# Install Code
COPY . /src

# Run
WORKDIR /src
# RUN pip install -e .

CMD python
