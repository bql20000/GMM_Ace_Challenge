import matplotlib.pyplot as plt
import numpy as np
import sklearn
from mpl_toolkits import mplot3d

def visualize2D(X):
    x = X[:, 0]
    y = X[:, 1]
    plt.scatter(x, y)
    plt.show()

def visualize3D(X):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, color='g')
    plt.show()