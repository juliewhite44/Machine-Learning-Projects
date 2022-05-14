import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("dane.data", delimiter='\t', header=None)

data = data.values

scaler = MinMaxScaler()
# transform data
data = scaler.fit_transform(data)

X = data[:, :-1]
Y = data[:, -1].reshape(data.shape[0], 1)


def base_function(X):
    Z = np.copy(X)
    for i in range(len(Z)):
        for j in range(len(Z[i])):
            Z[i][j] = Z[i][j]**2
    return Z


def model(X, Y, learning_rate, epochs, reg_lambda):
    m = Y.size
    theta = np.zeros((X.shape[1], 1))
    for i in range(epochs):
        y_pred = np.dot(base_function(X), theta)
        d_theta = (1 / m) * np.dot(X.T, y_pred - Y)
        # l1 reg
        # for i in range(len(d_theta)):
            # d_theta[i] += reg_lambda * np.sign(theta[i])
        theta = theta - learning_rate * d_theta

    return theta


epochs = 100
learning_rate = 0.05

# Searching the best lambda for L1

"""
lambda_list = [0.01 * i for i in range(100)]
cost_list = []
for _lambda in lambda_list:
    theta = model(X_train, Y_train, learning_rate, epochs, _lambda)

    y_pred = np.dot(X_val, theta)
    # cost
    # cost = (1 / (2 * Y_test.size)) * np.sum(np.square(y_pred - Y_test))

    # l1 cost
    # cost = (1 / (2 * Y_test.size)) * (np.sum(np.square(y_pred - Y_test)) + sum(np.abs(t) for t in theta))
    cost = (1 / (2 * Y_val.size)) * (np.sum(np.square(y_pred - Y_val)) + sum(np.abs(t) for t in theta))
    cost_list.append(cost)

best_lambda = cost_list.index(min(cost_list))*0.01
print(min(cost_list), cost_list.index(min(cost_list)))
"""
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

    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.67, random_state=random_state)

    X = np.vstack((np.ones((X.shape[0],)), X.T)).T

    X_train = np.vstack((np.ones((X_train.shape[0],)), X_train.T)).T
    X_test = np.vstack((np.ones((X_test.shape[0],)), X_test.T)).T
    X_val = np.vstack((np.ones((X_val.shape[0],)), X_val.T)).T

    for size in sizes:
        if size != 1:
            _, X_train_frac, _, Y_train_frac = train_test_split(X_train, Y_train, test_size=size,
                                                                random_state=random_state)
        else:
            X_train_frac = X_train
            Y_train_frac = Y_train
        best_lambda = 0.03

        theta = model(X_train_frac, Y_train_frac, learning_rate, epochs, best_lambda)
        y_pred = np.dot(X_test, theta)

        cost = (1 / (2 * Y_test.size)) * np.sum(np.square(y_pred - Y_test))
        print("Cost is: ", cost)

        results[size].append(cost)

for size in sizes:
    print('average for size', size, 'equals', sum(results[size]) / (len(results[size])))

points = [sum(results[size]) / (len(results[size])) for size in sizes]

X = np.array(sizes)
Y = np.array(points)
# Plotting point using scatter method
f, ax = plt.subplots(1)
plt.ylabel("cost")
plt.xlabel("size of training data")
plt.scatter(X, Y)
n = [0.01, 0.02, 0.03, 0.125, 0.625, 1]
for i, txt in enumerate(n):
    plt.annotate(txt, (X[i], Y[i]))
plt.plot(X, Y)
plt.show()