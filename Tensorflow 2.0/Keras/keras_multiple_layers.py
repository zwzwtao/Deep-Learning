import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("tensorflow version: {}".format(tf.__version__))

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 6, 9, 12, 18])

plt.scatter(x, y)

# select all the lines(using :), and all the columns except the first column and the last column(1:-1)
# x = data.iloc[:, 1:-1]
# select the last column
# y = data.iloc[:, -1]

# now we directly define layers in the initial function
# the second layer don't need to specify input_shape any more
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(1,), activation="relu"),
                             tf.keras.layers.Dense(1)])

model.compile(optimizer="adam", loss="mse")
model.fit(x, y, epochs=100)
prediction = model.predict(np.array([12, 3]))
print(prediction)
