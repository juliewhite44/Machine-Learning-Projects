import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def predict(X_train, X_test, Y_train, k):
    m = X_train.shape[0]
    n = X_test.shape[0]
    y_pred = []

    for i in range(n):
        distance = []
        for j in range(m):
            d = (np.sqrt(np.sum(np.square(X_test[i, :] - X_train[j, :]))))
            distance.append((d, Y_train[j]))
        distance = sorted(distance)

        neighbors = []
        for item in range(k):
            neighbors.append(distance[item][1][0])

        y_pred.append(1 if sum(neighbor == 1 for neighbor in neighbors) > sum(neighbor == -1 for neighbor in neighbors) else -1)
    return y_pred


data = pd.read_csv("phishing.data", delimiter=',', header=None)

data = data.values

X = data[:, :-1]
Y = data[:, -1].reshape(data.shape[0], 1)


results = {
    0.01: [],
    0.02: [],
    0.03: [],
    0.125: [],
    0.625: [],
    1: []
}
sizes = [0.01, 0.02, 0.03, 0.125, 0.625, 1]

for random_state in range(3):
    print('random_state', random_state)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random_state)

    X_train = np.vstack((np.ones((X_train.shape[0],)), X_train.T)).T
    X_test = np.vstack((np.ones((X_test.shape[0],)), X_test.T)).T

    for size in sizes:
        print('size', size)
        if size != 1:
            _, X_train_frac, _, Y_train_frac = train_test_split(X_train, Y_train, test_size=size,
                                                                random_state=random_state)
        else:
            X_train_frac = X_train
            Y_train_frac = Y_train

        y_pred = predict(X_train_frac, X_test, Y_train_frac, 16)

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





