import numpy as np

# define RNN parameters, here we only have two time steps: t1 and t2
# t1=1, t2=2
X = [1, 2]
# initial state
state = [0.0, 0.0]
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# forward propagation
for i in range(len(X)):
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    # update state
    state = np.tanh(before_activation)
    final_output = np.dot(state, w_output) + b_output
    print("before activation:", before_activation)
    print("state:", state)
    print("output:", final_output)
