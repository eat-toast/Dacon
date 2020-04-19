from fbprophet import Prophet
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import mean_squared_error


if 'Mission_AIFrenz_Season1' not in os.getcwd():
    os.chdir(os.path.join(os.getcwd(), 'Mission_AIFrenz_Season1'))# 디렉토리 변경
data_path =  os.path.join(os.getcwd(), 'data')

train = pd.read_csv(os.path.join(data_path, 'train.csv'))
test = pd.read_csv(os.path.join(data_path, 'test.csv'))


ts_days_idx = pd.date_range('2017-01-01', periods=4320)
df_ts = pd.DataFrame(range(len(ts_days_idx))
                     , columns=['y']
                     , index=ts_days_idx)

sensor_list = list(train.columns[41:60])

df_ts['y'] = train.iloc[:-432].Y17.values
df_ts = df_ts.reset_index()
df_ts.columns = ['ds', 'y']

m = Prophet(changepoint_prior_scale=5)
m.fit(df_ts)

future = m.make_future_dataframe(periods=144*3)
forecast = m.predict(future.iloc[-144*3:])
forecast2 = m.predict(future)
# fig1 = m.plot(forecast2)



fig, ax = plt.subplots(1, 1, figsize=(15,7))
ax.plot(forecast2)
ax.plot(forecast.yhat.values)
ax.plot(train.iloc[-432:].Y18.values)
plt.tight_layout()

tt = train.iloc[-432:].Y18.rolling(window = 10).mean()
tt.plot()


tt=  train.iloc[-432:].Y18
tt = tt.reset_index(drop=True)

np.argmin( tt.loc[:4450]) # 4352
np.argmax( tt.loc[:4450]) # 4396

76-32
173-76
np.argmin( tt.iloc[100:200]) # 173
np.argmax( tt.iloc[173:300]) # 217
217-173
322-217
np.argmin( tt.iloc[217:400]) # 322
np.argmax( tt.iloc[322:]) # 359
359-322

train = train.iloc[33:].reset_index(drop = True)

for i in range(0, 4751, 44):
    print(i)