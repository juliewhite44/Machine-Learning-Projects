import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def train(x_train, y_train):
    y_probability = [0, 0]
    for i in range(len(y_train)):
        y_probability[y_train[i]] += 1

    x_probability = [{}, {}]
    for c in range(2):
        for i in range(9):
            for k in range(1, 11):
                x_probability[c][(i, k)] = (len([ind for ind in range(len(x_train)) if x_train[ind][i] == k and
                                                 y_train[ind] == c]) + 1) / (y_probability[c] + 10)

    y_probability[0] += 1
    y_probability[1] += 1
    y_probability[0] /= len(y_train) + 2
    y_probability[1] /= len(y_train) + 2

    return x_probability, y_probability


def predict(x_test, x_probability, y_probability):
    predictions = []
    for x in x_test:
        zero_probability = x_probability[0][(0, x[0])] * x_probability[0][(1, x[1])] * x_probability[0][(2, x[2])] * \
                           x_probability[0][(3, x[3])] * \
                           x_probability[0][(4, x[4])] * x_probability[0][(5, x[5])] * x_probability[0][(6, x[6])] * \
                           x_probability[0][(7, x[7])] * \
                           x_probability[0][(8, x[8])] * y_probability[0]
        one_probability = x_probability[1][(0, x[0])] * x_probability[1][(1, x[1])] * x_probability[1][(2, x[2])] * \
                          x_probability[1][(3, x[3])] * \
                          x_probability[1][(4, x[4])] * x_probability[1][(5, x[5])] * x_probability[1][(6, x[6])] * \
                          x_probability[1][(7, x[7])] * \
                          x_probability[1][(8, x[8])] * y_probability[1]

        if zero_probability > one_probability:
            predictions.append(0)
        else:
            predictions.append(1)

    return predictions


data = pd.read_csv("rp.data", delim_whitespace=True, header=None)

x = data.iloc[:, :-1].values
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
    x_zero_train, x_zero_test, y_zero_train, y_zero_test = train_test_split(x_zero, y_zero, test_size=1 / 3,
                                                                            random_state=g)
    x_one_train, x_one_test, y_one_train, y_one_test = train_test_split(x_one, y_one, test_size=1 / 3, random_state=g)

    x_train = x_zero_train + x_one_train
    x_test = x_zero_test + x_one_test
    y_train = y_zero_train + y_one_train
    y_test = y_zero_test + y_one_test

    for h in errors.keys():
        x_probability, y_probability = train(x_train[:int(np.floor(len(x_train) * h))],
                                             y_train[:int(np.floor(len(y_train) * h))])
        predictions = predict(x_test, x_probability, y_probability)
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
    err[i] = sum(errors[h]) / 10

_, M = plt.subplots(1)
M.set_ylim(ymin=0, ymax=0.25)

plt.scatter(H, err)
plt.plot(H, err)
plt.show()

# 0.01 0.22982456140350874
# 0.02 0.14342105263157895
# 0.03 0.12192982456140351
# 0.125 0.06842105263157894
# 0.625 0.051315789473684204
# 1 0.023245614035087715
