import tensorflow as tf
'''
add the regularization to collection to make the codes more readable
The reason we want to store the regularization of weights of each layer to 
the collection is that we can easily add them up when we computing the final 
loss fuction.
'''

# get weights from one layer, and then add the L2_regularization
# of this weights to a collection called 'losses'
def get_weight(shape, lambd):
    '''
    returns the parameter(weight) after doing L2 regularization on it
    '''
    var = tf.Variable(tf.random_normal(shape, dtype=tf.float32))
    # we define a collection called 'losses' and add the L2_regularization of 'var' to it
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
    return var

# assume two inputs
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
# define the number of nodes in each layer(here we have 5 layers)
layer_dimension = [2, 10, 10, 10, 1]
# number of layers
n_layers = len(layer_dimension)

# current layer
cur_layer = x
# the number of nodes of current layer, initialized as layer 0's
in_dimension = layer_dimension[0]

# now we generate a five-layer fully connected neural network
for i in range(1, n_layers):
    next_dimension = layer_dimension[i]
    # generate the weight matrix of current layer and add the L2_regularized
    # weights to collection 'losses'
    weight = get_weight([in_dimension, next_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[next_dimension]))
    # using ReLU activation
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    in_dimension = layer_dimension[i]

# now we already have all the L2_regularized weight loss in collection
# whild doing forward propagation, so we only have to compute the simple loss
# of training result
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# add the mean square loss to losses collection
tf.add_to_collection('loss', mse_loss)

# now we can simply add all the elements in collection to get the final loss function
loss = tf.add_n(tf.get_collection('losses'))

