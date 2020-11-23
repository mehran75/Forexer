from RNN import SimpleRNN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import yaml


with open("configuration/parameters.yaml", 'r') as stream:
    parameters = yaml.safe_load(stream)


dataset_path = parameters['']
label = input('Chosen Label')
time_window =  