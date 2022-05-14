from random import random
import numpy as np
import sys

A = (-8, -3)
B = (-5, 10)
C = (4, 2)

all_points = []


def gen_points_in_circle(s, r):  # generate 500 random points in circle s - center and r - radius
    for i in range(500):
        d = random() * r
        angle = random() * 2 * np.pi
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        result = (s[0] + x, s[1] + y)
        all_points.append(result)


def gen_points_out_circle(s, r):  # generate 10 random points outside the circle, (0,1) from circumference
    for i in range(10):
        d = random() + r
        angle = random() * 2 * np.pi
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        result = (s[0] + x, s[1] + y)
        all_points.append(result)


gen_points_in_circle(A, 3.5)
gen_points_in_circle(B, 4.2)
gen_points_in_circle(C, 2.1)

gen_points_out_circle(A, 3.5)
gen_points_out_circle(B, 4.2)
gen_points_out_circle(C, 2.1)

original_stdout = sys.stdout

with open('circ.csv', 'w') as f:  # all points to csv file
    sys.stdout = f
    print('X,Y')
    for k in all_points:
        print(k[0], k[1], sep=',')
    sys.stdout = original_stdout
