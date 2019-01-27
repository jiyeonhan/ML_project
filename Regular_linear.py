##############################################
# Regularized Linear Model example :
#
# 1) Ridge regression : Hyperparameter, alpha, determines the penalty term, 0.5*alpha*sum_i=1 w_i^2 
#                       The large alpha reduces any model variance (flatter), but increases its bias
#                       LinearRegression(SGDRegression) with penalty=l2
# 2) Lasson regression : Huperparameter, alpha, determines the penalty term, alpha*sum_i=1 w_i
#                        It tends to eliminate the least important features  (set them to zero).
#                        It automatically performs feature selections and output s a sparse model 
#                       (with few non zero feature weights)
#                        LinearRegression(SGDRegression) with penalty=l1
# 3) Elastic Net : Combination of Ridge and Lasso depending on r, penalty term = r*Lasso_term + (1-r)*Ridge_term
#
# * Early stopping : you just stop training as soon as the validation error reaches the minimum.
#                   The algorithm learns and its prediction error (RMSE) on the training set natually goes down, and so does its prediction error on the validation set.
#                   (to reduce overfitting possibility => RMSE of training keeps descreasing with more Epoches)
#
#
##############################################

import sklearn
#from sklearn.model_selection import train_test_split                                                                        
import numpy as np
import matplotlib.pyplot as plt


X = 2 * np.random.rand(100,1)
y = 4 + 3*X + np.random.randn(100,1)

#X_b = np.c_[np.ones((100,1)), X]

#Regression using Ridge regression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV

alpha_set = [0.1*i for i in range(10)]
reg = RidgeCV(alphas=alpha_set, cv=len(alpha_set))
reg.fit(X, y)
print("Best alpha = ", reg.alpha_)

reg_ridge = Ridge(alpha = reg.alpha_, solver='cholesky') # another option for solver, 'auto'
reg_ridge.fit(X, y)
print("Coefficient = ", reg_ridge.coef_ , " : Intercept = ", reg_ridge.intercept_)
print("Score for Ridge = ", reg_ridge.score(X,y))

from sklearn.linear_model import Lasso
reg_lasso = Lasso(alpha=reg.alpha_)
reg_lasso.fit(X, y)
reg_lasso.predict([[1.5]])

from sklearn.linear_model import ElasticNet
reg_elnet = ElasticNet(alpha=reg.alpha_, l1_ratio=0.5)
reg_elnet.fit(X, y)
reg_elnet.predict([[1.5]])

plt.figure()
plt.scatter(X,y)
plt.plot(X, reg_ridge.intercept_ + reg_ridge.coef_*X, 'r-')
plt.plot(X, reg_lasso.intercept_ + reg_lasso.coef_*X, 'g-')
plt.plot(X, reg_elnet.intercept_ + reg_elnet.coef_*X, 'b-')
plt.legend(['Ridge', 'Lasso', 'ElasticNet'])
plt.text(0.0, 9.0, 'alpha = %s' % str(reg.alpha_))
plt.show()
