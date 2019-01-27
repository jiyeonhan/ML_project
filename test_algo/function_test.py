import numpy as np
import matplotlib.pyplot as plt

"""
def calfunc(x, gamma, dif):
    d = np.multiply(np.ones((len(x),1)), dif)                    
    print x.shape, d.shape
    y = np.exp(np.multiply((x - d)**2, -1.0*gamma))

    return y
"""
def f(t, gam, dif):
    return np.exp(gam*(x - dif)**2)

x = np.arange(-3.0, 3.0, 0.1).reshape(-1,1)
#yout1 = calfunc(x, 0.3, 1.0)
#yout2 = calfunc(x, 1.0, 1.0)

plt.figure(1)
plt.plot(x, f(x, -0.3, 1.0), 'r')
plt.plot(x, f(x, -1.0, 1.0), 'b')
plt.text(-2.0, 0.2, 'gamma = 0.3, l = 1.0', bbox=dict(facecolor='red', alpha=0.5))
plt.text(-2.0, 0.1, 'gamma = 1.0, l = 1.0', bbox=dict(facecolor='blue', alpha=0.5))
plt.show()
