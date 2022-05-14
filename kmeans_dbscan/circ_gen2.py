from random import random
import numpy as np
import sys

all_points = []


def gen_points_in_circle(r):  # generate circle with 500 points 0, 0 - center, r - radius
    for i in range(500):
        d = random()/4 + r
        angle = random() * 2 * np.pi
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        result = (x, y)
        all_points.append(result)


gen_points_in_circle(2)
gen_points_in_circle(5)
gen_points_in_circle(8)
gen_points_in_circle(11)

original_stdout = sys.stdout

with open('circ2.csv', 'w') as f:  # points to csv file
    sys.stdout = f
    print('X,Y')
    for k in all_points:
        print(k[0], k[1], sep=',')
    sys.stdout = original_stdout
