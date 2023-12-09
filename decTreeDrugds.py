import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import tensorflow_decision_forests as tfdf

df = pd.read_csv('/kaggle/input/drugs-a-b-c-x-y-for-decision-trees/drug200.csv')
df.head()

train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_dataset, label="Drug")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_dataset, label="Drug")

model = tfdf.keras.RandomForestModel(verbose=2)
model.fit(train_ds)
model.compile(metrics=["accuracy"])
print(model.evaluate(train_ds))
print(model.evaluate(test_ds))

tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0,max_depth=5)

columns=(['Age'],['Sex'],['BP'],['Cholesterol'],['Na_to_K'],['Drug'])
x = pd.DataFrame({
    "Age": [30],
    "Sex": ['F'],
    "BP": ['LOW'],
    "Cholesterol": ['HIGH'],
    "Na_to_K": [12]
})
x=tfdf.keras.pd_dataframe_to_tf_dataset(x)
print(type(x))
model.predict(x)

model_2 = tfdf.keras.GradientBoostedTreesModel(verbose=2)

model_2.compile(metrics=["accuracy"])
model_2.fit(train_ds, validation_data=test_ds)
print(model_2.evaluate(test_ds, return_dict=True))

x = pd.DataFrame({
    "Age": [30],
    "Sex": ['F'],
    "BP": ['LOW'],
    "Cholesterol": ['HIGH'],
    "Na_to_K": [12]
})
x=tfdf.keras.pd_dataframe_to_tf_dataset(x)
print(type(x))
model_2.predict(x)

model_3 = tfdf.keras.GradientBoostedTreesModel(
    num_trees=500, growing_strategy="BEST_FIRST_GLOBAL", max_depth=20)
model_3.compile(metrics=["accuracy"])
model_3.fit(train_ds, validation_data=test_ds)
print(model_3.evaluate(test_ds, return_dict=True))

x = pd.DataFrame({
    "Age": [30],
    "Sex": ['F'],
    "BP": ['LOW'],
    "Cholesterol": ['HIGH'],
    "Na_to_K": [12]
})
x=tfdf.keras.pd_dataframe_to_tf_dataset(x)
print(type(x))
model_3.predict(x)

model_4 = tfdf.keras.GradientBoostedTreesModel(
    num_trees=500,
    growing_strategy="BEST_FIRST_GLOBAL",
    max_depth=20,
    split_axis="SPARSE_OBLIQUE",
    categorical_algorithm="RANDOM",
    )
model_4.compile(metrics=["accuracy"])
model_4.fit(train_ds, validation_data=test_ds)
print(model_4.evaluate(test_ds, return_dict=True))

x = pd.DataFrame({
    "Age": [30],
    "Sex": ['F'],
    "BP": ['LOW'],
    "Cholesterol": ['HIGH'],
    "Na_to_K": [12]
})
x=tfdf.keras.pd_dataframe_to_tf_dataset(x)
print(type(x))
model_4.predict(x)