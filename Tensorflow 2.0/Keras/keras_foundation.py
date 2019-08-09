import tensorflow as tf

print("tensorflow version: {}".format(tf.__version__))

x = [1, 2, 3, 4, 6]
y = [3, 6, 9, 12, 18]

model = tf.keras.Sequential()
# the first param of Dense is the output size(unit), the batch size is 5, it's already auto batched
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model.summary()

# use default learning rate of adam optimizer
model.compile(optimizer="adam", loss="mse")

history = model.fit(x, y, epochs=5000)

# print(model.predict(x)) will output:
# [[ 4.986885]
#  [ 7.106471]
#  [ 9.226057]
#  [11.345642]
#  [15.584814]]
