import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def sig(t):
    return 1 / (1 + np.exp(-t))


def train(x_train, y_train, learning_rate, epochs):
    theta = np.zeros(len(x_train[0]))
    for i in range(epochs):
        x_train, y_train = shuffle(np.array(x_train), np.array(y_train))
        for j in range(len(x_train)):
            res = learning_rate * (y_train[j] - sig(np.dot(theta.T, x_train[j])))
            for k in range(len(theta)):
                theta[k] += res * x_train[j][k]
    return theta


def predict(x_test, theta):
    return [1 if sig(np.dot(theta.T, k)) > 0.5 else 0 for k in x_test]


data = pd.read_csv("rp.data", delim_whitespace=True, header=None)

x = data.iloc[:, :-1].values
x = (x - x.mean()) / x.std()
y = data.iloc[:, -1].values

y = [0 if i == 2 else 1 for i in y]

y_zero = [i for i in y if i == 0]
y_one = [i for i in y if i == 1]

x_zero = [k for i, k in enumerate(x) if y[i] == 0]
x_one = [k for i, k in enumerate(x) if y[i] == 1]

errors = {
    0.01: [],
    0.02: [],
    0.03: [],
    0.125: [],
    0.625: [],
    1: []
}

for g in range(10):
    x_zero_train, x_zero_test, y_zero_train, y_zero_test = train_test_split(x_zero, y_zero, test_size=1 / 3, random_state=g)
    x_one_train, x_one_test, y_one_train, y_one_test = train_test_split(x_one, y_one, test_size=1 / 3, random_state=g)

    x_train = x_zero_train + x_one_train
    x_test = x_zero_test + x_one_test
    y_train = y_zero_train + y_one_train
    y_test = y_zero_test + y_one_test

    for h in errors.keys():
        theta = train(x_train[:int(np.floor(len(x_train) * h))], y_train[:int(np.floor(len(y_train) * h))], 0.003, 10)
        predictions = predict(x_test, theta)
        bad_pred = 0
        for i in range(len(predictions)):
            if predictions[i] != y_test[i]:
                bad_pred += 1
        errors[h].append(bad_pred / len(y_test))

H = np.zeros(6)
err = np.zeros(6)

for i, h in enumerate(errors.keys()):
    print(h, sum(errors[h]) / 10)
    H[i] = h
    err[i] = sum(errors[h])/10

_, M = plt.subplots(1)
M.set_ylim(ymin=0, ymax=0.1)

plt.scatter(H, err)
plt.plot(H, err)
plt.show()


# 0.01 0.035526315789473684
# 0.02 0.03508771929824561
# 0.03 0.03333333333333334
# 0.125 0.0337719298245614
# 0.625 0.03771929824561403
# 1 0.02894736842105263



