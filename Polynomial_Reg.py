import sklearn
#from sklearn.model_selection import train_test_split                                                                        
import numpy as np
import matplotlib.pyplot as plt

"""
Polynomial Regression
"""

## Produce data sample
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

### Define pre feature using 2nd degree polynomial function
from sklearn.preprocessing import PolynomialFeatures
poly_feature = PolynomialFeatures(degree=2, include_bias = False)

X_poly = poly_feature.fit_transform(X)  # fit data, X, based on polynomial function and then transfer to X_poly

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()  # X_poly already has 2nd polynomial feature -> fit linear regression
lin_reg.fit(X_poly, y)
print("intercept = ", lin_reg.intercept_, "linear regression coefficient = ", lin_reg.coef_)

def f(t):
    return lin_reg.intercept_ + lin_reg.coef_[0][0]*t + lin_reg.coef_[0][1]*t**2

t = np.arange(-3.0, 3.0, 0.1)
 
plt.figure(1)
plt.subplot(211)
#plt.scatter(X, y)
plt.plot(X)
plt.plot(X_poly)
plt.subplot(212)
plt.scatter(X, y)
plt.plot(t, f(t), 'r--')
plt.show()

## Test and validation the result

from sklearn.metrics import mean_squared_error # MSE loss function
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

train_error, val_error = [], []
for m in range(1, len(X_train)):
    lin_reg.fit(X_train[:m], y_train[:m])
    y_train_pred = lin_reg.predict(X_train[:m])
    train_error.append(mean_squared_error(y_train_pred, y_train[:m]))
    y_val_pred = lin_reg.predict(X_val)
    val_error.append(mean_squared_error(y_val_pred, y_val))

plt.figure(2)
plt.xlabel('Training set size')
plt.ylabel('RMSE')
plt.plot(np.sqrt(train_error), 'r-+', linewidth=2, label='train')
plt.plot(np.sqrt(val_error), 'b-', linewidth=2, label='eval')
plt.legend(['train', 'val'], loc='best')
plt.show()

## Present the model using Pipeline

from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline((
        ("poly_regression", PolynomialFeatures(degree=2, include_bias=False)),
        ("lin_reg_conv", LinearRegression()),
        ))

polynomial_regression.fit(X, y)
#print(lin_reg_conv.intercept_, lin_reg_conv.coef_)
print("prediction for x=0.0 is ", polynomial_regression.predict([[0.0]]))
print("Score = ", polynomial_regression.score(X, y))
print("Parameters info = ", polynomial_regression.get_params())

### Getting the coefficient from the fit in each step : model.named_steps['step_name'].coef_
print("fit parameters for linear regression = ", 
      polynomial_regression.named_steps['lin_reg_conv'].coef_,
      polynomial_regression.named_steps['lin_reg_conv'].intercept_)
      
