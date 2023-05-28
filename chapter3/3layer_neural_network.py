import numpy as np

# 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 출력층에 사용하는 활성화 함수 (항등 함수)
def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # Layer 1
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    # Layer 2
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    # Layer 3
    z3 = np.dot(z2, W3) + b3
    y = identity_function(z3) # output

    return y

network = init_network()
output = forward(network, np.array([1.0, 0.5]))

print(output)