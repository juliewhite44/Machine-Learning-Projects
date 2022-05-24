import sys
from math import exp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def sigmoid(x):
    for i in range(len(x[0])):
        x[0][i] = 1 / (1 + exp(-x[0][i]))
    return x


def dsigmoid(sig):
    for i in range(len(sig[0])):
        sig[0][i] * (1 - sig[0][i])
    return sig


def loss(y, y_pred):
    return 2 * (y_pred - y) / y.size


class Dense:
    def __init__(self, input_size, output_size):
        self.input = None
        self.w = np.random.rand(input_size, output_size) - 0.5
        self.b = np.random.rand(1, output_size) - 0.5

    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.w) + self.b

    def backward(self, error, lr):
        input_error = np.dot(error, self.w.T)
        weights_error = np.dot(self.input.T, error)

        self.w -= lr * weights_error
        self.b -= lr * error
        return input_error


class Sigmoid:
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return sigmoid(self.input)

    def backward(self, error, _):
        return dsigmoid(self.input) * error


class Network:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input_data):
        result = []

        for i in range(len(input_data)):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        for i in range(epochs):
            for j in range(len(x_train)):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)

                error = loss(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            print('epoch', i + 1)


data = pd.read_csv("phishing.data", delimiter=',', header=None)

data = data.values

X = data[:, :-1]
Y = data[:, -1].reshape(data.shape[0], 1)
for i in range(len(Y)):
    if Y[i][0] == -1:
        Y[i][0] = 0

epochs = 50
learning_rate = 0.001

results = {
    0.01: [],
    0.02: [],
    0.03: [],
    0.125: [],
    0.625: [],
    1: []
}
sizes = [0.01, 0.02, 0.03, 0.125, 0.625, 1]

for random_state in range(5):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random_state)

    for size in sizes:
        if size != 1:
            _, X_train_frac, _, Y_train_frac = train_test_split(X_train, Y_train, test_size=size,
                                                                random_state=random_state)
        else:
            X_train_frac = X_train
            Y_train_frac = Y_train

        X_train_frac = np.reshape(X_train_frac, (X_train_frac.shape[0], 1, X_train_frac.shape[1]))
        Y_train_frac = np.reshape(Y_train_frac, (Y_train_frac.shape[0], 1, 1))
        net = Network([Dense(30, 15), Sigmoid(), Dense(15, 1), Sigmoid()])
        net.fit(X_train_frac, Y_train_frac, epochs=50, learning_rate=0.01)
        y_pred = net.predict(X_test)
        y_pred = np.array(y_pred)
        y_pred = np.reshape(y_pred, y_pred.shape[0])
        for i in range(len(y_pred)):
            if y_pred[i] < 0.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        error = 1 - accuracy_score(Y_test, y_pred)

        results[size].append(error)

for size in sizes:
    print('average for size', size, 'equals', sum(results[size]) / (len(results[size])))

points = [sum(results[size]) / (len(results[size])) for size in sizes]

print(results)

X = np.array(sizes)
Y = np.array(points)

f, ax = plt.subplots(1)
plt.ylabel("error")
plt.xlabel("size of training data")
plt.scatter(X, Y)
n = [0.01, 0.02, 0.03, 0.125, 0.625, 1]
for i, txt in enumerate(n):
    plt.annotate(txt, (X[i], Y[i]))
plt.plot(X, Y)
plt.show()

# x_good, y_good, x_phishing, y_phishing = read_data(do_standardize=False)
# X_train, X_test, y_train, y_test = split_data(x_good, y_good, x_phishing, y_phishing, random_state=2)
#
# X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# y_train = np.reshape(y_train, (y_train.shape[0], 1, 1))
#
# net = Network([Dense(30, 15), Sigmoid(), Dense(15, 1), Sigmoid()])
#
# net.fit(X_train, y_train, epochs=50, learning_rate=0.01)
#
# y_pred = net.predict(X_test)
# y_pred = np.array(y_pred)
# y_pred = np.reshape(y_pred, y_pred.shape[0])
# for i in range(len(y_pred)):
#     if y_pred[i] < 0.5:
#         y_pred[i] = 0
#     else:
#         y_pred[i] = 1
#
# good_prediction = len([i for i in range(len(y_test)) if y_pred[i] == y_test[i]])
# accuracy = good_prediction / len(y_test) * 100
# print(accuracy)
