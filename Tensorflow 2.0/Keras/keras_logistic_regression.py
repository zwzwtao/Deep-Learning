import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# if there is no header in the raw datafile, then add
# "header=None", this will add index for the columns and thus have the first index line
data = pd.read_csv("dataset/credit-a.csv", header=None)

# value_count = data.iloc[:, -1].value_counts()
# print(value_count)
# 1    357
# -1    296

x = data.iloc[:, :-1]
# a = data.iloc[1, :-1]
# print(a)
# replace -1 from the dataset to 0, since we want to have 0 and 1, not -1 and 1
y = data.iloc[:, -1].replace(-1, 0)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, input_shape=(15,), activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["acc"]                 # metrics is optional, if specified, will output a list(?) that contains accuracy etc.
)

history = model.fit(x, y, epochs=100)

print(history.history.keys())

plt.plot(history.epoch, history.history.get('loss'))
plt.show()

print(model.predict(data.iloc[0:2, :-1]))














