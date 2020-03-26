import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

os.chdir(os.path.join(os.getcwd(), 'Mission_AIFrenz_Season1'))# 디렉토리 변경
data_path =  os.path.join(os.getcwd(), 'data')

train = pd.read_csv(os.path.join(data_path, 'train.csv'))
test = pd.read_csv(os.path.join(data_path, 'test.csv'))

# 데이터 설명
# -     대전지역에서 측정한 실내외 19곳의 센서데이터와, 주변 지역의 기상청 공공데이터를 semi-비식별화하여 제공합니다.
# -     센서는 온도를 측정하였습니다.
# -     모든 데이터는 시간 순으로 정렬 되어 있으며 10분 단위 데이터 입니다.
# -     예측 대상(target variable)은 Y18입니다.
#
#
# train.csv
# -     30일 간의 기상청 데이터 (X00~X39) 및 센서데이터 (Y00~Y17)
# -     이후 3일 간의 기상청 데이터 (X00~X39) 및 센서데이터 (Y18)
#
# test.csv
# -     train.csv 기간 이후 80일 간의 기상청 데이터 (X00~X39)



temperature_name = ["X00","X07","X28","X31","X32"] #기압
localpress_name  = ["X01","X06","X22","X27","X29"] #현지기압
speed_name       = ["X02","X03","X18","X24","X26"] #풍속
water_name       = ["X04","X10","X21","X36","X39"] #일일 누적강수량
press_name       = ["X05","X08","X09","X23","X33"] #해면기압
sun_name         = ["X11","X14","X16","X19","X34"] #일일 누적일사량
humidity_name    = ["X12","X20","X30","X37","X38"] #습도
direction_name   = ["X13","X15","X17","X25","X35"] #풍향

# Y18번의 데이터를 나머지들의 평균값으로 채운다. (GeonwooKim 님 코드)
t = train["Y18"].isna()
null_index = t[t==True].index
train.loc[null_index, "Y18"] = train.loc[null_index, "Y00":"Y17"].mean(axis=1)

# null값을 채워주고, 데이터의 구조를 살핀다. (뚱냥이 님 코드)
print('total number of sample in train :',train.shape[0])
print('total number of Y18 in train :',train.shape[0] - train['Y18'].isnull().sum())


# id 변수를 삼각함수를 이용해 시간 변수 추가 (26님 코드 -- 기상 캐스터 잔나)
minute = (train.id%144).astype(int)
hour= pd.Series((train.index%144/6).astype(int))

minute_test = (test.id%144).astype(int)
hour_test = pd.Series((test.index%144/6).astype(int))


min_in_day = 24*6
hour_in_day = 24

minute_sin = np.sin(np.pi*minute/min_in_day)
minute_cos = np.cos(np.pi*minute/min_in_day)

hour_sin  = np.sin(np.pi*hour/hour_in_day)
hour_cos  = np.cos(np.pi*hour/hour_in_day)

t1 = range(len(minute_sin[:144]))
plt.plot(t1, minute_sin[:144],
         t1, minute_cos[:144],
         t1, train.loc[:143, 'Y18'], 'r-')
plt.title("Sin & Cos")
plt.show()

# 그림을 보니, sin이 더 맞는 것 같아 sin 추가
train['minute_sin'] = minute_sin
train['hour_sin'] = hour_sin
minute_sin_test = np.sin(np.pi*minute_test/min_in_day)
hour_sin_test  = np.sin(np.pi*hour_test/hour_in_day)

test['minute_sin'] = minute_sin_test
test['hour_sin'] = hour_sin_test



# data-set 구조 바꾸기 (뚱냥이 님 코드)
# panel data
# 데이터셋을 panel 형태로 바꾸어보면 어떨까요. 각각의 센서 변수들 Y00~Y18 을 panel 형태로 바꾸어보겟습니다.

# sensor list
sensor_list = list(train.columns[41:60])

# panel dataset
df = pd.melt(train,
        id_vars='id',
        value_vars=list(train.columns[41:60]), var_name='sensor')

df = pd.merge(df,train.drop(columns=sensor_list), on='id' ) # 합치기.
df = df.dropna() # 결측제거.
idx = df.value > 0
df = df.loc[idx]
df.head()

df.loc[df.sensor=='Y00', 'value'].rolling(window=6, min_periods=1).mean()

from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals()) # sql ready!
qry = '''
select 
from df
group by sensor
'''
user_claim_cnt = sqldf(qry)

####################
X_train = df.loc[:, df.columns[3:]]
y_train = df.loc[:, 'value']

df.sensor=='Y18


X_test = test.iloc[:, 1:]

lgb_train = lgb.Dataset(X_train, label=y_train)

# custom metric -- 대회 목적에 맞는 방법으로 변경 (자카종신 님 코드)
def mse1(y_pred, dataset):
    y_true = dataset.get_label()

    diff = abs(y_true - y_pred)
    less_then_one = np.array([0 if x < 1 else 1 for x in diff])

    y_pred = less_then_one * y_pred
    y_true = less_then_one * y_true

    score = mean_squared_error(y_true, y_pred)

    return 'score', score, False

# GeonwooKim님 코드
lgb_param = {
    "objective":"regression",
    "learning_rate":0.001
}

print("cv start")
cv_result = lgb.cv(
    lgb_param,
    lgb_train,
    feval=mse1,
    num_boost_round=99999,
    nfold=5,
    early_stopping_rounds=100,
    stratified=False,
    verbose_eval=10
)

print("train start")
lgb_model = lgb.train(
    lgb_param,
    lgb_train,
    num_boost_round=len(cv_result["l2-mean"])
)

pred = lgb_model.predict(X_test)

#######
# 제출
submission = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
submission['Y18'] = pred
submission.to_csv('submit/lgb_base_line3_20200923.csv',index = False)