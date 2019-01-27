import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()

print(list(iris.keys()))

#['target', 'DESCR', 'target_names', 'feature_names', 'data', 'filename']

#X = iris['data'][:, 3:] #petal width
#y = (iris['target'] == 2).astype(np.int) # 1 if Iris.Virginica, else 0

name = iris['target_names']

print("Name = ", name)

x0 = iris['data'][iris['target']==0]
x1 = iris['data'][iris['target']==1]
x2 = iris['data'][iris['target']==2]                 
 
plt.figure(1)
plt.subplot(2,2,1)
n, bins, patch = plt.hist(x0[:,0], 10, range=(0.0, 10.0), color='green', alpha=0.75)
n, bins, patch = plt.hist(x1[:,0], 10, range=(0.0, 10.0), color='red', alpha=0.75)
n, bins, patch = plt.hist(x2[:,0], 10, range=(0.0, 10.0), color='blue', alpha=0.75)
plt.xlabel('Feature 1')
plt.legend([name[0], name[1], name[2]], loc='best')

#plt.hist(x1[:,0], bins='20', 'r')
#plt.hist(x2[:,0], bins='20', 'g')
plt.subplot(2,2,2)
n, bins, patch = plt.hist(x0[:,1], 10, range=(0.0, 10.0), color='green', alpha=0.75)
n, bins, patch = plt.hist(x1[:,1], 10, range=(0.0, 10.0), color='red', alpha=0.75)
n, bins, patch = plt.hist(x2[:,1], 10, range=(0.0, 10.0), color='blue', alpha=0.75)
plt.xlabel('Feature 2')
plt.subplot(2,2,3)
n, bins, patch = plt.hist(x0[:,2], 10, range=(0.0, 10.0), color='green', alpha=0.75)
n, bins, patch = plt.hist(x1[:,2], 10, range=(0.0, 10.0), color='red', alpha=0.75)
n, bins, patch = plt.hist(x2[:,2], 10, range=(0.0, 10.0), color='blue', alpha=0.75)
plt.xlabel('Feature 3')
plt.subplot(2,2,4)
n, bins, patch = plt.hist(x0[:,3], 10, range=(0.0, 10.0), color='green', alpha=0.75)
n, bins, patch = plt.hist(x1[:,3], 10, range=(0.0, 10.0), color='red', alpha=0.75)
n, bins, patch = plt.hist(x2[:,3], 10, range=(0.0, 10.0), color='blue', alpha=0.75)
plt.xlabel('Feature 4')
#plt.show()
plt.savefig('iris_feature.png')

##Logistic only returns the class which has the best probablity 

from sklearn.linear_model import LogisticRegression

def feature_sel(feature, target):
    
    x = iris['data'][:, feature:feature+1]    
    y = iris['target']
    #y = (iris['target'] == target).astype(np.int)

    logic = LogisticRegression(multi_class='multinomial', solver='lbfgs') #C : inverse of regularization strength (smaller value for stronger regularization)

    logic.fit(x, y)
    print("x, y shape = ", x.shape, y.shape)
    print("y value = ", y)

    X_new = np.linspace(0, 10, 1000).reshape(-1,1)
    y_prob = logic.predict_proba(X_new)

    return X_new, y_prob



xl=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']

#print("y_prediction = ", y_out)
plt.figure(2)
for i in range(4):
    xval, yval = feature_sel(i, 0)
    plt.subplot(2,2,i+1)
    plt.plot(xval, yval[:,0], 'g-')  
    plt.plot(xval, yval[:,1], 'r-')  
    plt.plot(xval, yval[:,2], 'b-')  
    plt.xlabel(xl[i])
    if i==0:
        plt.legend([name[0], name[1], name[2]], loc='best')
plt.savefig('iris_logistic_reg.png')
plt.show()
