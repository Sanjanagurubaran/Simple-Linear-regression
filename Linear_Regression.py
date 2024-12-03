"""Linear Regression Model for icecream sales with respect to temperature"""


import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
temperature=[20,25,30,35,40]
icesales=[13,21,25,35,38]
X=np.array([temperature]).T
""" X denoting the temperature (independent variable)"""
Y=np.array(icesales)
"""Y denoting icecream sales(dependent variable)"""
rmodel=LinearRegression()
rmodel=rmodel.fit(X,Y)
"""Model for Linear Regression"""
Y_predict=rmodel.predict(X)
"""Predicted Y """
plt.scatter(temperature,icesales,marker='*',edgecolors='r')
plt.plot(temperature,Y_predict,'-bo')
plt.show()
"""Graphical Representation"""
