# 라이브러리 환경
import pandas as pd
import numpy as np

# sklearn 테이터셋에서 iris 데이터셋 로딩
from sklearn import datasets
iris = datasets.load_iris()

# iris 데이터셋은 딕셔너리 형태이므로, key 값 확인
'''
print(iris.keys())
print(iris['DESCR'])
print("데이터 셋 크기:", iris['target'])
print("데이터 셋 내용:\n", iris['target'])
'''

# data 속성의 데이터셋 크기
print("데이터 셋 크기:", iris['data'].shape)

# data 속성의 데이터셋 내용(첫 7개 행 추출)
data1 = ['a', 'b', 'c', 'd', 'e']
print(type(data1))
sr1 = pd.Series(data1)
# print(type(sr1))
data2 = (1, 2, 3.14, 100, -10)
sr2 = pd.Series(data2)

dict_data = {'c1':data1, 'c2':data2}
df = pd.DataFrame(dict_data)
print(df)


# 열(columns)과 행(index)이름 바꾸기
df.columns = ['string1', 'string2']
df.index = ['r1', 'r2', 'r3', 'r4', 'r5']

# print(df.loc['r2':'r4', 'string1':'string2'])

print('데이터셋 내용:\n', iris['data'][:7, :])
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

print('데이터 프레임의 형태:', df.shape)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print(df.head(2))

df['Target'] = iris['target']
print(df.head())

x = [2, 1, 13, 4, 15, 26]
y = [0, 4, 31, 2, 42, 54]

df = pd.DataFrame({'X':x, 'Y':y})
print(df)






















