import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)


def model(X, y, learning_rate=0.001, epochs=1000, l=0.01):
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for _ in range(epochs):
        for idx, x in enumerate(X):
            if y[idx] * (np.dot(x, w) + b) >= 1:
                dw = l * w
                db = 0
            else:
                dw = l * w - np.dot(y[idx][0], x)
                db = -y[idx]
            w -= learning_rate * dw
            b -= learning_rate * db
    return w, b


data = pd.read_csv("phishing.data", delimiter=',', header=None)

data = data.values

X = data[:, :-1]
Y = data[:, -1].reshape(data.shape[0], 1)

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

    X_train = np.vstack((np.ones((X_train.shape[0],)), X_train.T)).T
    X_test = np.vstack((np.ones((X_test.shape[0],)), X_test.T)).T

    for size in sizes:
        if size != 1:
            _, X_train_frac, _, Y_train_frac = train_test_split(X_train, Y_train, test_size=size,
                                                                random_state=random_state)
        else:
            X_train_frac = X_train
            Y_train_frac = Y_train

        w, b = model(X_train_frac, Y_train_frac, learning_rate, epochs)
        y_pred = predict(X_test, w, b)

        error = 1-accuracy_score(Y_test, y_pred)

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