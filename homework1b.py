import numpy as np


X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([5, 8, 11, 14, 17], dtype=float)


m = np.cov(X, Y, bias=True)[0][1] / np.var(X)
c = np.mean(Y) - m * np.mean(X)

print("Slope (m):", m)
print("Intercept (c):", c)
print("Prediction for x=6:", m*6 + c)
#output
#Slope (m): 3.0
#Intercept (c): 2.0
#Prediction for x=6: 20.0
