import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x_train = [[25],[36],[39],[45],[61],[70]] 
y_train = [[16],[18],[20],[26],[30],[34]] 

x_test = [[23],[30],[42],[39],[62],[70]] 
y_test = [[15],[17],[20],[26],[30],[33]] 

regressor = LinearRegression()
regressor.fit(x_train,y_train)
xx = np.linspace(0,70,200)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx,yy)

model_degree_featurizer = PolynomialFeatures(degree=2)

x_train_quad = model_degree_featurizer.fit_transform(x_train)
y_train_quad = model_degree_featurizer.transform(x_test)

reg_quad = LinearRegression()
reg_quad.fit(x_train_quad,y_train)
xx_quad = model_degree_featurizer.transform(xx.reshape(xx.shape[0],1))

plt.plot(xx, reg_quad.predict(xx_quad),c='r',linestyle='--')
plt.title('Percentage value increase of crates of stock as per crate stored')
plt.xlabel('Number of crates of stock in warehouse')
plt.ylabel('Percentage increase in value of crates')
plt.axis([0,75,0,35])
plt.grid(True)
plt.scatter(x_train,y_train)
plt.show()
print(x_train)
print(x_train_quad)
print(y_train)
print(y_train_quad)

