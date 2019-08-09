import tensorflow as tf
import numpy as np
# import keras


# lstm = tf.keras.layers.LSTMCell(1000)
# state = lstm.zero_state(100, tf.float32)

# loss = 0
# for i in range(num_steps):
#     if i > 0: tf.get_variable_scope().reuse_variables()
#
#     lstem_output, state = lstm(current_input, state)
#     final_output = fully_connected(lstm_output)
#
#     loss += calc_loss(final_output, expected_output)

HIDDEN_SIZE = 30  # LSTM中隐藏节点的个数。
NUM_LAYERS = 2  # LSTM的层数。
TIMESTEPS = 10  # 循环神经网络的训练序列长度。
TRAINING_STEPS = 10000  # 训练轮数。
BATCH_SIZE = 32  # batch大小。
TRAINING_EXAMPLES = 10000  # 训练数据个数。
TESTING_EXAMPLES = 1000  # 测试数据个数。
SAMPLE_GAP = 0.01  # 采样间隔。




def generate_data(seq):
    X = []
    y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入；第i + TIMESTEPS项作为输
    # 出。即用sin函数前面的TIMESTEPS个点的信息，预测第i + TIMESTEPS个点的函数值。
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    # print(X)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# X = generate_data(np.linspace(0,10.01,100))

cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

print(np.sin([1, 2, 3]))



test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
print(test_start)
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(
    0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

print(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

outputs, _ = tf.nn.dynamic_rnn(cell, train_X, dtype=tf.float32)
output = outputs[:, -1, :]

print(len(train_X))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(outputs[1]))















