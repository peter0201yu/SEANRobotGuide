#!/bin/bash

set -e
set -x

/usr/bin/python3 -m pip install --upgrade pip
pip3 install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install --upgrade --force-reinstall sklearn \
  numpy \
  tensorboard \
  pandas \
  jupyterlab \
  ipympl \
  tqdm \
  click \
  seaborn \
  matplotlib \
  theseus-ai
