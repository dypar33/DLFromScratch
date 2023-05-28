import numpy as np
import pickle
from mnist import load_mnist

# 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 출력층에 사용하는 활성화 함수 (소프트맥스 함수)
def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)

    return exp_a / sum_exp_a

# 미리 정의된 가중치로 수행
def init_network():
    with open('./sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
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
    y = softmax(z3) # output

    return y

def load_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)

    return x_test, t_test

x, t = load_data()
network = init_network()
accuracy_value = 0

batch_size = 100

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_value += np.sum(p == t[i:i+batch_size])

print("Accuracy: {}".format(float(accuracy_value) / len(x)))