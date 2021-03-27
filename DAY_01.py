import pandas as pd

'''
print(pd.__version__) # 판다스 버전 확인
'''
data1 = ['a', 'b', 'c', 'd', 'e']
'''
print(data1)
print("자료형:", type(data1))


srl = pd.Series(data1)
print("자료형:", type(srl))
print(srl)

# 객체의 원소 추출
print(srl.loc[0])
print(srl.loc[1:3]) # .loc는 1부터 3까지 모든 원소를 출력해줌
print(srl[1:3]) # 1이상 3미만인 인덱스의 값만 출력해줌

# data2 객체 확인
data2 = (1, 2, 3.14, 100, -10) # tuple 형태
sr2 = pd.Series(data2)
print(sr2)
'''

# 1차원 구조의 시리즈를 여러 개 결합 : dict 사용
data1 = ['a', 'b', 'c', 'd', 'e']
sr1 = pd.Series(data1)

data2 = [1, 2, 3.14, 100, -10]
sr2 = pd.Series(data2)

dict_data = {'c0':sr1, 'c1':sr2}
df1 = pd.DataFrame(dict_data)
print(df1)
print(type(df1))
print(df1.columns)

# 데이터 프레임 열속성 이름 변경 : df1.columns = ['name1', 'name2']
df1.columns = ['string', 'number']
print(df1)

# 행 인덱스 - 정수 0부터 4까지 오름차순으로 자동 지정된다.
print(df1.index)

# 행 인덱스 새로운 속성으로 변경
df1.index = ['r0', 'r1', 'r2', 'r3', 'r4']
print(df1)

# 데이터프레임의 일부분을 추출 : loc[행 인덱스, 열 이름]
print(df1.loc['r2', 'number'])

# 데이터프레임 인덱스 범위지정
print(df1.loc['r2':'r3', 'string':'number'])
print(df1.loc['r2':'r3', 'number'])
print(df1.loc['r2', 'string':'number'])
print(df1.loc[:, 'string'])
print(df1.loc['r2':'r3', :])