'''
x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-7, 61, -23, 7, -1, 43, -5, -11, -51, -29]

import pandas as pd
df = pd.DataFrame({'X':x, 'Y':y})
print(df.shape)

train_features = ['X']
target_cols = ['Y']
x_train = df.loc[:, ['X']]
y_train = df.loc[:, ['Y']]
print(x_train, y_train)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

print("기울기 : ", lr.coef_[0][0])
print("y절편 : ", lr.intercept_[0])

import numpy as np
x_new = np.array([11]).reshape(1, 1)
print(lr.predict(x_new))

x_data = np.arange(11, 16, 1).reshape(-1, 1)
print(lr.predict(x_data))
'''


x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-7, 61, -23, 7, -1, 43, -5, -11, -51, -29]

import matplotlib.pyplot as plt

'''
plt.plot(x, y)
plt.show()
'''

import pandas as pd
df = pd.DataFrame({'X':x, 'Y':y})
print(df.shape)

# df.head()
# df.tail()

train_features = ['X']
train_cols = ['Y']
x_train = df.loc[:, train_features]
y_train = df.loc[:, train_cols]

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
print(lr.coef_[0][0])
print(lr.intercept_[0])

import numpy as np
x_new = np.array([11]).reshape(1, 1)
print(lr.predict(x_new))


y1_new = np.arange(11, 16, 1).reshape(-1, 1)
y2_new = np.arange(11, 16, 1).reshape(1, -1)

print(y1_new.shape)
print(y2_new.shape)
print(lr.predict(y1_new))


from sklearn import datasets
iris = datasets.load_iris()
print(iris['DESCR'])
print(iris)