import os
import sys
import numpy as np
import pandas as pd
from six.moves import urllib
from pprint import pprint

# Hide the warnings (hooray)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load the titanic dataset
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Base Feature Columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ["age", "fare"]

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# Convert the input data and feed it into a pipeline
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

        if shuffle:
            ds = ds.shuffle(1000)
        
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds

    return input_function

# Create training and evaluation inputs
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Inspect the data sheet
ds = make_input_fn(dftrain, y_train, batch_size=10)()
for feature_batch, label_batch in ds.take(1):
    print("Feature keys:", list(feature_batch.keys()))
    print()
    print("Batch class:", feature_batch['class'].numpy())
    print()
    print("Batch labels:", label_batch.numpy())

# Train the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
print(probs)