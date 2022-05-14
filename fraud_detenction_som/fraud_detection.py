import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, colorbar, pcolor, plot, show

dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values  # X - all columns without last one
Y = dataset.iloc[:, -1].values  # Y - last column, credit approved or not

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

som = MiniSom(x=10, y=10, input_len=15)

som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

bone()
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']  # red circles -> customers who didn't get approval
colors = ['r', 'g']  # green square -> customers who got  approval
tab = np.zeros((10, 10))

for i, x in enumerate(X):  # loop over customer database, for each customer vector
    w = som.winner(x)  # getting the winning node for the particular customer
    if Y[i] == 1:
        tab[w[0]][w[1]] = 1  # = 1 if there is green square
    plot(w[0] + 0.5,  # x coordinate of winning node = w[0]
         w[1] + 0.5,  # y coordinate of the winning node = w[1], adding 0.5 to put marker in middle of square
         markers[Y[i]],  # link customer approval and markers
         markeredgecolor=colors[Y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)

first_white = 0
first_ind = (-1, -1)
second_white = 0
second_ind = (-1, -1)

for i in range(len(som.distance_map().T)):
    for j in range(len(som.distance_map().T[i])):
        if tab[j][i] == 1:
            if som.distance_map().T[i][j] >= first_white:
                second_white = first_white
                second_ind = first_ind
                first_white = som.distance_map().T[i][j]
                first_ind = (j, i)
            elif som.distance_map().T[i][j] >= second_white:
                second_white = som.distance_map().T[i][j]
                second_ind = (j, i)

print(first_ind)
print(second_ind)

show()

# finding frauds
mappings = som.win_map(X)  # key - winning square, value - client list of this square

frauds = np.concatenate((mappings[first_ind], mappings[second_ind]))
frauds = sc.inverse_transform(frauds)  # inverse to the original values

print('Suspected customers')
for i in frauds[:, 0]:
    print(int(i))
