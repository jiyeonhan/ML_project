from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def make_meshgrid(x, y, h=.02):
    """ Create a mesh of points to plot

    Parameters
    -----------
    x: data to base x-axis mashgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -----------
    xx, yy : ndarray
    """
    x_min, x_max = x.min()-1, x.max()+1
    y_min, y_max = y.min()-1, y.max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """ Plot the decision boundaries for a classifier
    
    Parameters
    ------------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out



iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

import matplotlib.pyplot as plt

x0 = [X[:,0][y==0], X[:,1][y==0]]
x1 = [X[:,0][y==1], X[:,1][y==1]]
x2 = [X[:,0][y==2], X[:,1][y==2]]

x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

from sklearn import neighbors

#n_neighbors = 15
n_neighbors = 3

plt.figure()
for weights, i in zip(['uniform', 'distance'], range(2)):
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights) # weights option = 'uniform', 'distance'
    clf.fit(X, y)
    
    plt.subplot(2,1,i+1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    
    plt.xlabel(iris.feature_names[2])
    plt.ylabel(iris.feature_names[3])
    
    plt.scatter(x0[0], x0[1], marker='^', c='g')
    plt.scatter(x1[0], x1[1], marker='o', c='b')
    plt.scatter(x2[0], x2[1], marker='s', c='r')
    print("score for ", weights, " = ", clf.score(X, y))
plt.show()



