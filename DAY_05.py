# 복습
import pandas as pd
x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-2, 32, -10, 5, 1, 23, -1, -4, -24, -13]
# Series : 시리즈 클래스는 1차원 배열 구조이다.
sr1 = pd.Series([1, 2, 3, 4, 5], index = ['a', 'b', 'c', 'd', 'e'])
# print(sr1)
# 엑셀로 보면 복수의 행(row)으로 이루어진 하나의 열(column)또는 하나의 행(row)으로 이루어진 복수의 열(column)의 값
df = pd.DataFrame({'X': x, 'Y': y})
# df = pd.DataFrame(x, y)
df.index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# print(df)
# print(df.X) # 지정 열(column) 확인
# print(df.loc['a']) # 지정 행(row) 확인
# print(df.loc['a':'c', :]) # 'a' 행부터 'c'행 까지 모든 열 출력

x_train = df.loc[:, ['X']] # 대괄호가 들어가야함
y_train = df.loc[:, ['Y']] # 대괄호가 들어가야함
# print(x_train.shape) # 10행 1열
# print(y_train.shape) # 10행 1열

# Modeling
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

# y = x + 1
# print("기울기 : ", lr.coef_[0][0])
# print("y절편 : ", lr.intercept_)

# 예측(predict) - 1
import numpy as np
X_new = np.array(11).reshape(1, 1)
# print("예측 값 : ", lr.predict(X_new))
# print("예측 값 형태 : ", lr.predict(X_new).shape)
# print("예측 값 타입 : ", type(lr.predict(X_new)))

# print()

# 예측(predict) - 2
X_data = np.arange(11, 16, 1).reshape(-1, 1)
# print("예측 값 : ", lr.predict(X_data))
# print("예측 값 형태 : ", lr.predict(X_data).shape)
# print("예측 값 타입 : ", type(lr.predict(X_data)))

# iris 데이터 셋 살펴보기
from sklearn import datasets
iris = datasets.load_iris()

## iris 데이터 셋은 딕셔너리 형태이므로, key 값 확인
# print(iris.keys())

## iris DESCR 키를 이용하여 데이터셋 설명(Description) 출력
# print(iris['DESCR'])

## target 속성의 데이터 셋 크기
# print(iris['target'].shape) # 150행

## target 속성의 데이터 셋 내용
# print(iris['target'])

## data 속성의 데이터 셋 크기
# print(iris['data'].shape) # 150행 4열

## data 속성의 데이터 셋 내용(첫 7개 행 추출)
# print(iris['data'])

## 데이터 프레임 변환
iris_data = pd.DataFrame(iris['data'], columns=iris['feature_names'])
# print(iris_data)
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] # 열 변환
iris_data['target'] = iris['target'] # iris_data에 새로운 열 추가
print('데이터 셋 크기 : ', iris_data.shape) # 150행 4열(1열 추가 됨)
print(iris_data.head())
