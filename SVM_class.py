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

from sklearn import datasets

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


# Test the classification without scaling each features
from sklearn.svm import LinearSVC

svc = LinearSVC()
svc.fit(x, y)
print("Score = ", svc.score(x, y))

#x_val = np.random.randint(0, 10, 4000).reshape(-1,4)
xval0 = np.random.normal(x0[:,0].mean(),x0[:,0].std(),100).reshape(-1,1)
xval1 = np.random.normal(x0[:,1].mean(),x0[:,1].std(),100).reshape(-1,1)
xval2 = np.random.normal(x0[:,2].mean(),x0[:,2].std(),100).reshape(-1,1)
xval3 = np.random.normal(x0[:,3].mean(),x0[:,3].std(),100).reshape(-1,1)

x_val = np.zeros((100,4))
for i in range(100):
    x_val[i][0] = xval0[i]
    x_val[i][1] = xval1[i]
    x_val[i][2] = xval2[i]
    x_val[i][3] = xval3[i]


pred = svc.predict(x_val)
print(len(x_val[:][pred==1]), len(pred[:][pred==1]))

plt.figure(2)
for i in range(4):
    plt.subplot(2,2,i+1)
    x0_sel = x_val[:, i][pred==0]
    x1_sel = x_val[:, i][pred==1]
    x2_sel = x_val[:, i][pred==2]
    n, bins, patch = plt.hist(x0_sel, 10, range=(0.0, 10.0), color='green', alpha=0.75)
    n, bins, patch = plt.hist(x1_sel, 10, range=(0.0, 10.0), color='red', alpha=0.75)
    n, bins, patch = plt.hist(x2_sel, 10, range=(0.0, 10.0), color='blue', alpha=0.75)

plt.show()
