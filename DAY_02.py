# 지도학습 vs 비지도학습
# machine learning process
'''
1. 문제파악
2. 데이터 탐색
3. 데이터 전처리
4. 모델 학습
5. 예측
'''

# 1. 문제파악
x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-4, 30, -12, 3, -1, 21, -3, -6, -26, -15]

# 2. 데이터 탐색
import matplotlib.pyplot as plt
# plt.plot(x, y)
# plt.show()

# 3. 데이터 전처리
import pandas as pd
df = pd.DataFrame({'X':x, 'Y':y})
print(df.shape) # 10행 2열
print(df.head()) # 첫 번째 부터 5개의 행 출력
print(df.tail()) # 뒤에서부터 5개의 행 출력

train_features = ['X']
target_cols = ['Y']
x_train = df.loc[:, ['X']]
y_train = df.loc[:, ['Y']]
print(x_train.shape, y_train.shape) # 10행 1열

# 4. 모델 학습
from sklearn.linear_model import LinearRegression
lr = LinearRegression() # 회귀 모델 정의
lr.fit(x_train, y_train) # 모델 학습

# 모델 : y = x - 1
print("lr.coef_ : ", lr.coef_[0][0]) # 기울기
print("lr.intercept_ : ", lr.intercept_[0]) # y절편

# 5. 예측
import numpy as np
x_new = np.array([11]).reshape(1, 1) # 학습에 사용한 x_train이 2차원 구조이므로, array함수로 11을 배열로 변환 후 reshape 메소드를 사용하여 (1행, 1열)형태의 2차원 구조로 변형한다.
print(lr.predict(x_new))

x_data = np.arange(11, 16, 1).reshape(-1, 1)
print(lr.predict(x_data))






















