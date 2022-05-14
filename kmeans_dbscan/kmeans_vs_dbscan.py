import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# read the data
outside_circles = pd.read_csv('circ.csv')
inside_circles = pd.read_csv('circ2.csv')

# scatter plot
plt.figure(figsize=(8, 4), dpi=100)
sns.scatterplot(data=outside_circles, x='X', y='Y')
plt.show()

# scatter plot
plt.figure(figsize=(8, 4), dpi=100)
sns.scatterplot(data=inside_circles, x='X', y='Y')
plt.show()


# define the function
def display_categories(model, data):
    labels = model.fit_predict(data)
    plt.figure(figsize=(8, 4), dpi=100)
    sns.scatterplot(data=data, x='X', y='Y', hue=labels, palette='Set1')
    plt.show()


# initiate the model
model = KMeans(n_clusters=3)
display_categories(model, outside_circles)
print(model.cluster_centers_)

# initiate the model
model = KMeans(n_clusters=4)
display_categories(model, inside_circles)
print(model.cluster_centers_)

# initiate the model
model = DBSCAN(eps=0.7)
display_categories(model, outside_circles)

# initiate the model
model = DBSCAN(eps=0.8)
display_categories(model, inside_circles)
