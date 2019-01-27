import sklearn
#from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

"""
Stochastic Gradient Descent :
 Gradient Descent choosing the instance randomly (algorithm is much less regular than normal Batch Gradient Descent)
 It is good to escape from local optima, but bad because it means that the algorithm can never settle at the min.
 Making the points with y = theta_0*x*2 + theta_1

Gradient Descent step : theta_next = theta - eta*Delta_theta MSE(theta)
"""

n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters

def learning_schedule(t):
    return t0/(t+t1)

#theta = np.random.randn(2,1) # random initialization
theta = np.random.randn(2,1) # random initialization
print("theta initialization = ", theta)

X = 2 * np.random.rand(100,1)
y = 4 + 3*X + np.random.randn(100,1)

X_b = np.c_[np.ones((100,1)), X]

#plt.figure()
#plt.scatter(X, y)
#plt.show()

m = 1000

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        #print epoch, i, gradients

from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter = 50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

X_new = np.array([[0],[2]])
y_pred = theta[0] + theta[1]*X_new
y_pred_SGD = sgd_reg.coef_ + sgd_reg.intercept_*X_new

print("theta = ", theta)
print(sgd_reg.intercept_, sgd_reg.coef_)
plt.plot(X_new, y_pred, "r-")
plt.plot(X_new, y_pred_SGD, "b-")
plt.plot(X, y, "b.")
plt.axis([0,2,0,15])
plt.show()
