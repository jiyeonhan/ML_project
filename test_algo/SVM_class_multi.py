######################################################################
# Support Vector Machine Classification / Regression
#
# boundary + margin : vectors should be away as far as possible from boundary and outside of margin for classification
# Regression is opposite (vectors should be within margin)
#
# It is important to scale each features to have better result, otherwise the margin is not extracted efficiently
######################################################################

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets

iris = datasets.load_iris()

print(list(iris.keys()))

#['target', 'DESCR', 'target_names', 'feature_names', 'data', 'filename']

#X = iris['data'][:, 3:] #petal width
#y = (iris['target'] == 2).astype(np.int) # 1 if Iris.Virginica, else 0
x = iris['data']
x0 = iris['data'][iris['target']==0]
x1 = iris['data'][iris['target']==1]
x2 = iris['data'][iris['target']==2]
y = iris['target']
                 
print(x0[:,0])
 
plt.figure(1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.subplot(2,2,1)
n, bins, patch = plt.hist(x0[:,0], 10, range=(0.0, 10.0), color='green', alpha=0.75)
n, bins, patch = plt.hist(x1[:,0], 10, range=(0.0, 10.0), color='red', alpha=0.75)
n, bins, patch = plt.hist(x2[:,0], 10, range=(0.0, 10.0), color='blue', alpha=0.75)
#plt.hist(x1[:,0], bins='20', 'r')
#plt.hist(x2[:,0], bins='20', 'g')
plt.subplot(2,2,2)
n, bins, patch = plt.hist(x0[:,1], 10, range=(0.0, 10.0), color='green', alpha=0.75)
n, bins, patch = plt.hist(x1[:,1], 10, range=(0.0, 10.0), color='red', alpha=0.75)
n, bins, patch = plt.hist(x2[:,1], 10, range=(0.0, 10.0), color='blue', alpha=0.75)

plt.subplot(2,2,3)
n, bins, patch = plt.hist(x0[:,2], 10, range=(0.0, 10.0), color='green', alpha=0.75)
n, bins, patch = plt.hist(x1[:,2], 10, range=(0.0, 10.0), color='red', alpha=0.75)
n, bins, patch = plt.hist(x2[:,2], 10, range=(0.0, 10.0), color='blue', alpha=0.75)

plt.subplot(2,2,4)
n, bins, patch = plt.hist(x0[:,3], 10, range=(0.0, 10.0), color='green', alpha=0.75)
n, bins, patch = plt.hist(x1[:,3], 10, range=(0.0, 10.0), color='red', alpha=0.75)
n, bins, patch = plt.hist(x2[:,3], 10, range=(0.0, 10.0), color='blue', alpha=0.75)


# Test the classification with various classification options

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

xin = iris['data'][:, :2]  # same with iris.data[:, :2]
yin = iris['target']

# create an instance of SVM with various options. Not scale the data yet
C = 1.0 # SVM regularization parameter (1/alpha, lower value implies higher penalty)
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))

models = (clf.fit(xin, yin) for clf in models)

#title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting
fig, sub = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = xin[:, 0], xin[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolor='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
