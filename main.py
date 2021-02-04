import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

# Load the titanic dataset
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv")
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Explore the data
print(dftrain.head())
print()
print(dftrain.describe())