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

# t1 = range(len(minute_sin[:144]))
# plt.plot(t1, minute_sin[:144],
#          t1, minute_cos[:144],
#          t1, train.loc[:143, 'Y18'], 'r-')
# plt.title("Sin & Cos")
# plt.show()

# 그림을 보니, sin이 더 맞는 것 같아 sin 추가



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

print(df.head())


from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from sklearn.ensemble import AdaBoostRegressor
from tsfresh.utilities.dataframe_functions import impute
from tqdm import tqdm

df_shift, y = make_forecasting_frame(train["Y18"], kind="temperature", max_timeshift=20, rolling_direction=1)

X = extract_features(df_shift, column_id="id", column_sort="time", column_value="value", impute_function=impute,
                     show_warnings=False, n_jobs = 0)

print(X.shape)
X = X.loc[:, X.apply(pd.Series.nunique) != 1]
print(X.shape)

X["feature_last_value"] = y.shift(1)
# Drop first line
X = X.iloc[1:, ]
y = y.iloc[1: ]

X.head()

ada = AdaBoostRegressor(n_estimators=10)
y_pred = [np.NaN] * len(y)

isp = 100  # index of where to start the predictions
assert isp > 0

for i in tqdm(range(isp, len(y))):
    ada.fit(X.iloc[(i-100):i], y[(i-100):i])
    y_pred[i] = ada.predict(X.iloc[i, :].values.reshape((1, -1)))[0]

y_pred = pd.Series(data=y_pred, index=y.index)



# Dataframe of predictions and true values
ys = pd.concat([y_pred, y], axis = 1).rename(columns = {0: 'pred', 'value': 'true'})

# Convert index to a datetime
ys.index = pd.to_datetime(ys.index)
ys.head()

ys.plot(figsize=(15, 8))
plt.title('Predicted and True Price')
plt.show()

# Create column of previous price
ys['y-1'] = ys['true'].shift(1)
ys[['y-1', 'true']].plot(figsize = (15, 8))
plt.title('Benchmark Prediction and True Price')
plt.show()



from tsfresh import extract_features


t = list(df.columns)
t.remove('value')
timeseries = df.loc[:, t]

extracted_features = extract_features(timeseries.head(100), column_id="sensor", column_sort="id", n_jobs=0)

from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

impute(extracted_features)
features_filtered = select_features(extracted_features, df.head(100).value, n_jobs=0)

import tsfresh
tt = tsfresh.feature_selection.relevance.calculate_relevance_table(extracted_features, df.head(100).value, ml_task='regression', n_jobs=0)

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
download_robot_execution_failures()
timeseries_, y = load_robot_execution_failures()


train['minute_sin'] = minute_sin
train['hour_sin'] = hour_sin
minute_sin_test = np.sin(np.pi*minute_test/min_in_day)
hour_sin_test  = np.sin(np.pi*hour_test/hour_in_day)

test['minute_sin'] = minute_sin_test
test['hour_sin'] = hour_sin_test
