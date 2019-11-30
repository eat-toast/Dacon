from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score


from scipy.stats import uniform, randint

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

import xgboost as xgb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
import warnings

warnings.filterwarnings('ignore')

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

from datetime import datetime
import featuretools as ft
from datetime import datetime
import os
from sklearn.ensemble import RandomForestClassifier
# from ggplot import *
from sklearn.linear_model import LinearRegression
import scipy.stats as st
# -*- coding: utf-8 -*-

### 데이터 불러오기 ###
dacon_data_path = os.path.join(os.getcwd(), 'Auction_master_kr')
os.chdir(dacon_data_path )

train=pd.read_csv("Auction_master_train.csv")
test=pd.read_csv("Auction_master_test.csv")
regist=pd.read_csv("Auction_regist.csv")
rent=pd.read_csv("Auction_rent.csv")
result=pd.read_csv("Auction_result.csv")

### 전처리 ###

#  Regist_date을 datetime 형태로 변환 (20160829 -> 2016-08-29)
regist_date = regist['Regist_date'].astype('str')
regist['Regist_date'] = pd.to_datetime(regist_date.str.slice(0,4) + "-" + regist_date.str.slice(4,6) + "-" + regist_date.str.slice(6,8), errors='coerce')

#  Auction_date와 Rent_date를 datetime 형태로 변환
result['Auction_date']=pd.to_datetime(result['Auction_date'])
rent['Rent_date']= pd.to_datetime(rent['Rent_date'],errors='coerce')


train= train.loc[train['Auction_key'] !=10,] # 아웃라이어 인가?


### target변수와의 상관관계 ###
train.corr()['Hammer_price'].sort_values()
#  Minimum_sales_price 0.992로 매우 높다. 감정을 한 가격과 실거래가가 매우 비슷한 상황.
#  실제로 이런 변수가 모델에 포함되면 다른 변수들이 학습이 잘 안된다.
# --> 예를들어, .. 실험을 해봐야 겠다.

# 산점도.
plt.scatter(train['Minimum_sales_price'],train['Hammer_price'])

#  여기서 문제를 비트는 아이디어 출현! #
#  Y 자체를 예측하는 것이 아닌, Hammer_price / Minimum_sales_price 의 비율(변화율)을 예측하는 문제로 바꾸었다.
#  마치 Y를 log(Y)로 바꾸는 것처럼 문제를 비틀었다. 마지막에 Minimum_sales_price 만 다시 곱해주면 되니까 가능하다.

### Min_sales_p로 나눈 것을 target으로 하는 모델 ###
train['real'] = train['Hammer_price'] / train['Minimum_sales_price']
test['real'] = test['Hammer_price'] / test['Minimum_sales_price']

# 시각화
# 출처 : https://python-graph-gallery.com/24-histogram-with-a-boxplot-on-top-seaborn/
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}) # sharex ? 무슨 옵션?

# Add a graph in each part
sns.boxplot(train['real'], ax=ax_box)
sns.distplot(train['real'], ax=ax_hist)

# 변수들의 상관관계를 본다.
train.corr()['real'].sort_values()

### EDA1 ###
#  Q1. 지역마다 real 의 차이가 있지 않을까?
#  Q2. 지역마다 최종 낙찰가격(Hammer_price)의 차이는?

#  2번은 당연히 서울이 부산보다 높은 가격일 것이다.
#  하지만, real 즉, 최저매각가격 대비 낙찰가의 변화율은 서울과 부산이 다를까?
#  아마, 각 지역별 노른자 땅이 변화율이 높지 않을까?
#  산간, 바다 지방보다는 도심에 가까운 지역이 real 값이 클 것이라 예상해 본다.

cop=train.copy() # train 데이터를 복사해 cop에 저장 --> 데이터에 변형을 마구마구 할 것이라 복사하는 것.
cop['x']=pd.cut(train['point.x'],100) # 100개의 구간으로 위경도를 자른다.
cop['y']=pd.cut(train['point.y'],100) # 100개의 구간으로 위경도를 자른다.

cop1=cop.groupby(['x','y']).size().reset_index(name='count') # 각 지역별 거래 횟수
cop2=cop.groupby(['x','y'])['real'].mean().reset_index(name='real') # 각 지역별 평균 변화율
cop3=cop.groupby(['x','y'])['Hammer_price'].mean().reset_index(name='Hammer_price') # 각 지역별 낙찰가격

cop=pd.merge(cop1,cop2,on=['x','y'],how='left')
cop=pd.merge(cop,cop3,on=['x','y'],how='left')


cop=cop.loc[cop['count']>=9] # 거래가 9건 이상인 데이터만 가져온다.
real = cop.pivot("y", "x", "real")
Hammer = cop.pivot("y", "x", "Hammer_price")
# plt.subplots(1, 1, figsize=(7, 5))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
sns.set(rc={'axes.facecolor':'black'})


sns.heatmap(real, annot=False, xticklabels=False,yticklabels=False,cmap="coolwarm", ax = ax1).set_title('real by area')
sns.heatmap(Hammer, annot=False, xticklabels=False,yticklabels=False,cmap="coolwarm", ax = ax2).set_title('Hammer_price by area')
plt.show()

#  A1. 지역마다 real 은 차이가 있다. 이 지역을 잘 나타내는 변수는 addr_dong(동) 정보가 있다.
len(train['addr_dong'].unique())
#  하지만 동은 285개나 존재한다...
#  여기서, 원 저자들은 위치 데이터를 보정할 수 있는 외부 데이터가 필요하다고 느낌.



### Feature Engineering 1 ###

#  가격 변화율 -> real 이 높다면, 좋은 물건이라 생각 할 수 있다.
#  그렇다면 안 좋은 물건은 절차상에 문제가 있을 수 있기 때문에 거래 기간이 오래 걸리지 않을까?
#  거래 기간에 따른 real의 변화를 살펴본다.

##  daydiff1 = 최종 경매일로부터 낙찰까지 걸린 일자
train['daydiff1']=(pd.to_datetime(train['Close_date'],errors='coerce') - pd.to_datetime(train['Final_auction_date'],errors='coerce')).dt.days
# mean encoding - daydiff1의 결측치는 평균값으로
train['daydiff1']= np.where(train['daydiff1'].isnull(),train['daydiff1'].mean(),train['daydiff1'])

##  datdiff = 최초 경매일 - 감정일자
train['daydiff']=(pd.to_datetime(train['First_auction_date'],errors='coerce') - pd.to_datetime(train['Appraisal_date'],errors='coerce')).dt.days
# 음수인 경우, 중앙값으로 대체.
train['daydiff']=np.where(train['daydiff']<0, train['daydiff'].median(), train['daydiff'])

##  daydiff2 = 최종 경매일 - 최초 경매일
train['daydiff2']=( pd.to_datetime(train['Final_auction_date'],errors='coerce')-pd.to_datetime(train['First_auction_date'],errors='coerce') ).dt.days

temp_col = ['daydiff', 'daydiff1', 'daydiff2', 'real']
train[temp_col].corr()['real'].sort_values()

plt.style.use('fivethirtyeight')
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

ax1.scatter(train['daydiff'], train['real']);ax1.set_title('daydiff')
ax2.scatter(train['daydiff1'], train['real']);ax2.set_title('daydiff1')
ax3.scatter(train['daydiff2'], train['real']);ax3.set_title('daydiff2')

# test에도 같은 절차 진행
test['daydiff1']=(pd.to_datetime(test['Close_date'],errors='coerce') - pd.to_datetime(test['Final_auction_date'],errors='coerce')).dt.days
test['daydiff1']=np.where(test['daydiff1'].isnull(),test['daydiff1'].mean(),test['daydiff1'])
test['daydiff']=(pd.to_datetime(test['First_auction_date'],errors='coerce') - pd.to_datetime(test['Appraisal_date'],errors='coerce')).dt.days
test['daydiff']=np.where(test['daydiff']<0,test['daydiff'].median(),test['daydiff'])
test['daydiff2']=(pd.to_datetime(test['Final_auction_date'],errors='coerce') - pd.to_datetime(test['First_auction_date'],errors='coerce')).dt.days


### Feature Engineering 2 ###

#  오래된 건물은 신식보다 가격 변동량(real)이 작지 않을까?
#  경매에서 새로운 건물을 사람들이 사고싶어 할테니 말이다.
#  regist(등기부등본)을 이용하여 가장 일찍 등기된 날짜가 건설작업완료 날짜의 추정치로 삼는다.


train['built_date'] = train.apply(lambda x: regist.loc[(regist['Auction_key']==x['Auction_key']),"Regist_date"].min(),axis=1)
train['built_date'] = pd.DatetimeIndex(train['built_date']).year
train['built_date'] = np.where(train['built_date'].isnull(),train['built_date'].median(),train['built_date'])
train['built_date1'] = train['built_date']<=2000


# test에도 같은 방식 적용
test['built_date']=test.apply(lambda x: regist.loc[(regist['Auction_key']==x['Auction_key']),"Regist_date"].min(),axis=1)
test['built_date']=pd.DatetimeIndex(test['built_date']).year
test['built_date']=np.where(test['built_date'].isnull(),test['built_date'].median(),test['built_date'])
test['built_date1']=test['built_date']<=2000

### Feature Engineering 3 ###

#  법원 경매는 물건을 무조건 판다. 어떻게?? 이번에 안팔리면 다음엔 20% 할인해서 판매한다.
#  그래서 이를 반영한 (Minimum_sales_price / Total_appraisal_price) = falld 라는 변수를 만든다.
#  --> 사실 이 부분은 이해가 잘 되지 않는다. 다시 나누고 곱하고... 이 변수가 도움이 되는지는 직접 실험을 해봐야겠다.

train['falld']=(train['Minimum_sales_price']/train['Total_appraisal_price']).round(3)

a=pd.Series([  0.134,0.168,0.21,0.262,0.328,0.41,0.512,0.64,0.8,1,1])
b=pd.Series([0,0.134,0.168,0.21,0.262,0.328,0.41,0.512,0.64,0.8,1,1])

train['lastupp']=train.apply(lambda x: a[b==x['falld']].iloc[0],axis=1)


test['falld']=(test['Minimum_sales_price']/test['Total_appraisal_price']).round(3)

a=pd.Series([0.134,0.168,0.21,0.262,0.328,0.41,0.512,0.64,0.8,1,1])
b=pd.Series([0,0.134,0.168,0.21,0.262,0.328,0.41,0.512,0.64,0.8,1,1])

test['lastupp']=test.apply(lambda x: a[b==x['falld']].iloc[0],axis=1)

train['upp']=train['lastupp']/train['falld']
test['upp']=test['lastupp']/test['falld']

train['lastupp_p']=train['lastupp']*train['Total_appraisal_price']
test['lastupp_p']=test['lastupp']*test['Total_appraisal_price']

### Feature Engineering 4 ###

#  유찰 횟수에 대한 변수
#  일반적으로 유찰횟수가 늘어나면 Hammer_price는 낮아진다. (왜? 사람들이 안사서 법원이 가격이 떨어뜨린 횟수를 의미하기 때문)
#  그렇다면 real 변수에 대해서는 어떻게 작동할까?
train['Auction_miscarriage_count'].value_counts().sort_index()

sns.factorplot('Auction_miscarriage_count','real', data =train)

#  유찰횟수가 0이면 (법원 경매에서 한번에 낙찰된 경우) real이 낮은 것을 볼 수 있다.
#  이를 위해 one이라는 변수를 새로 만든다.
train['one']=train['Auction_miscarriage_count']==0
test['one']=test['Auction_miscarriage_count']==0


### Feature Engineering 5 ###

#  당신이 살고 있는 아파트는 어떤 브랜드인가?
#  법원 경매도 이름이 있는 아파트라면 가격이 더 올라가지 않을까? 아파트 브랜드 변수를 만들자.
#  만약 5가지 중 어느 곳에도 포함되어 있지 않는다면 other로 대체한다.
apartname=pd.Series(["더샵","자이","아이파크","래미안","힐스테이트"])

# 이 함수는 어떻게 동작하는지 ..
def get_apartname(string_val,apartname):
    if apartname.apply(lambda x: [x in y for y in [string_val]][0]).sum()>0:
        return np.asscalar(apartname.loc[apartname.apply(lambda x: [x in y for y in [string_val]][0])])
    else:
        return "other"

train['apart']=train.apply(lambda x: get_apartname(string_val=x['addr_etc'],apartname=apartname),axis=1)
test['apart']=test.apply(lambda x: get_apartname(string_val=x['addr_etc'],apartname=apartname),axis=1)

sns.factorplot('apart','real',data=train)

### Feature Engineering 6 ###

#  Creditor라는 categorical variable은 너무 category 가 많고,
#  개별 카테고리에 해당하는 sample 수도 적기 때문에, 묶어줄 필요가 있다.
#  Creditor에서 가장 흔한 5가지 종류 이외에 것들은 그냥 other로 처리하기로 했다.

namecred=pd.Series(train['Creditor'].value_counts().head().index.tolist())


def getcred(string_val, namecred):
    if namecred.isin([string_val]).sum() > 0:
        return np.asscalar(namecred.loc[namecred.isin([string_val])])
    else:
        return "other"


train['cred'] = train.apply(lambda x: getcred(x['Creditor'], namecred=namecred), axis=1)
test['cred'] = test.apply(lambda x: getcred(x['Creditor'], namecred=namecred), axis=1)

sns.factorplot('cred','real',data=train)

### Feature Engineering 7 ###

#  건물주가 되려는 당신... 하지만 그 속에 이미 누군가 살고 있다면 보증금 문제가 발생할 수 있다.
#  나는 건물만 삿는데, 이미 살고 있는 사람들의 보증금까지 물어줘야 될 수 있다.
#  이런 문제를 다루는 키워드가 말소기준권리 라고 한다.
#  한마디로 정리하면, 세입자와 법원 중 누가 먼저 살고 있었는지 판가름 하는 것이다.
#  만약, 세입자가 먼저 살고 있었다면 보증금을 물어줘야 할 수 있다. (세입자의 대항력)
#  그러니, 경매에서 집을 구매하려는 사람은 말소기준권리를 잘 따져봐야 한다


##  경매에서 권리의 인수여부는 말소기준권리를 기준으로 결정된다.
#   말소기준권리는 저당,압류, 가압 중에서 가장 먼저 일어난 것이며,
#   각각의 Auction_key마다 이것을 찾아 malsogijun이라는 데이터프레임을 만들었다.

regist['Regist_class']=regist['Regist_class'].astype('str').str.replace(pat=" ",repl="").astype('str')
rent['Rent_class']=rent['Rent_class'].astype('str').str.replace(pat=" ",repl="").astype('str')
rent['Rent_date']=pd.to_datetime(rent['Rent_date'],errors='coerce')

idx = pd.Series(regist['Regist_class'] ).isin(["저당","압류","가압"]) # 저당, 압류, 가압 찾기
malsogijun = regist.loc[idx, ].groupby(['Auction_key'])['Regist_date'].min().reset_index()

#  이 말소기준보다 전에 일어나고 점유하고 있는 임차인(대항력을 가진 임차인)을 찾아
#  해당 경매의 낙찰자가 인수해야 할 Rent_deposit을 take_over_rent라는 변수에 저장한다

def get_rent_malso(Rent_date,Auction_key,Rent_class):
    #if pd.isnull(Rent_date) or len((malsogijun.loc[malsogijun['Auction_key']==Auction_key,"Regist_date"]))==0:
    #    return np.nan
    #else:
    return (#np.asscalar((Rent_date<malsogijun.loc[malsogijun['Auction_key']==Auction_key,"Regist_date"]).values) &
                           (Rent_class=="점유"))


k=rent.apply(lambda x: get_rent_malso(Rent_date=x['Rent_date'],Auction_key=x['Auctiuon_key'],Rent_class=x['Rent_class']),axis=1)

rent['k']=k
k = rent.loc[rent['k']==True,].groupby(['Auctiuon_key'])['Rent_deposit'].sum().reset_index() # 보증금 계산하기
k.columns=['Auction_key','take_over_rent']
train=pd.merge(train,k,how='left')
test=pd.merge(test,k,how='left')

train['take_over_rent'] = np.where(train['take_over_rent'].isnull(),0,train['take_over_rent'])
test['take_over_rent'] = np.where(test['take_over_rent'].isnull(),0,test['take_over_rent'])

#  등기부등본 상에서도 말소기준 권리 이전에 일어난 모든 권리행위의 Regist_price를 더해 take_over_regist라는 변수를 만든다.
def get_regist_malso(Regist_date, Auction_key, Regist_class):
    if pd.isnull(Regist_date) or len((malsogijun.loc[malsogijun['Auction_key'] == Auction_key, "Regist_date"])) == 0:
        return np.nan
    else:
        return (np.asscalar(
            (Regist_date < malsogijun.loc[malsogijun['Auction_key'] == Auction_key, "Regist_date"]).values) &
                (pd.Series(Regist_class).isin(["전세권", "임차권", "가등기"])))


k = regist.apply(lambda x: get_regist_malso(Regist_date=x['Regist_date'], Auction_key=x['Auction_key'],
                                            Regist_class=x['Regist_class']), axis=1)

regist['k'] = k
k = regist.loc[regist['k'] == True,].groupby(['Auction_key'])['Regist_price'].sum().reset_index()
k.columns = ['Auction_key', 'take_over_regist']
train = pd.merge(train, k, how='left')
test = pd.merge(test, k, how='left')

train['take_over_regist'] = np.where(train['take_over_regist'].isnull(), 0, train['take_over_regist'])
test['take_over_regist'] = np.where(test['take_over_regist'].isnull(), 0, test['take_over_regist'])

##  이제 이 둘을 합쳐 낙찰자가 감내 해야될 모든 금액을 구한다.
train['take_over_sum']=train['take_over_regist']+train['take_over_rent']
test['take_over_sum']=test['take_over_regist']+test['take_over_rent']
sns.distplot(train['take_over_sum'])

# 이렇게 take_over_rent와 take_over_regist라는 변수를 더해서 take_over_sum이라는 변수,즉 낙찰자가 인수해야 하는 모든 금액을 가리키는 변수를 만든다
# (그림에서 보다시피 이 변수도 분포가 sparse하다. take_over_sum이 0보다 큰 경우가 매우 적기 때문이다. 실제로 이번 대회에서는 이런 sparse함 때문에
# 오히려 이 변수를 넣는 것이 점수를 떨어뜨리지만, 데이터가 조금만 더 많아서 xgboost와 같은 부스팅 모델을 쓴다면 매우 중요한 변수가 될 것이므로 실제 모델에서는 빼지 않기로 했다)

##  나중에 이 변수를 제거해보고 실험을 해봐야겠다.


### Feature Engineering 8 ###

##  집을 구매할때 은행 2곳에 돈을 빌렸다면? 그 사람들에게 돈을 갚아야 할 것이다.
##  badang (배당) 이라는 변수는 돈을 받을 사람의 수를 구하는 변수 같다.
## 잠시 생각해보니, 돈을 받을 사람의 수와 아파트 경매 가격과 무슨 상관이 있나 싶다.

a = regist.apply(lambda x: pd.Series(x['Regist_class']).isin(['전세권', '가등기', '임차권']), axis=1)
a.columns = ['a']

regist1 = regist.loc[a['a'],]


def get_badang(Auction_key, a=regist1):
    return len(a.loc[a['Auction_key'] == Auction_key,])


train['badang'] = train.apply(lambda x: get_badang(x['Auction_key'], a=regist1), axis=1)
test['badang'] = test.apply(lambda x: get_badang(x['Auction_key'], a=regist1), axis=1)

sns.factorplot('badang','real',data=train)


### Feature Engineering 9 ###

#  train과 test에 있는 Auction_key 중에서 rent와 regist에 없는 key도 있다. 이것을 is_rent와 is_regist라고 변수화시킴.
train['is_rent']=train.apply(lambda x: pd.Series(x['Auction_key']).isin(list(rent['Auctiuon_key'])),axis=1)
test['is_rent']=test.apply(lambda x: pd.Series(x['Auction_key']).isin(list(rent['Auctiuon_key'])),axis=1)

train['is_regist']=train.apply(lambda x: pd.Series(x['Auction_key']).isin(list(regist['Auction_key'])),axis=1)
test['is_regist']=test.apply(lambda x: pd.Series(x['Auction_key']).isin(list(regist['Auction_key'])),axis=1)

sns.factorplot('is_rent','real',data=train)
sns.factorplot('is_regist','real',data=train)

# regist나 rent에 대한 정보가 있는 것이 오히려 real값이 적어지는 경향이 있는 것을 알 수 있다.


### Feature Engineering 10 ###
#  해당 방의 층을 아파트의 전체 층으로 나눈 층의 비율을 floor_rate 라는 변수로 만듦.
#  아파트마다 층이 다르므로, 이를 비교하기위한 normalization 과정이라 할 수 있다.

##  대회 기간중 실제로 같은 변수를 만들어 회귀모델에 넣었을때는 별 효과를 보지 못했었다.
train['floor_rate']=train['Current_floor']/train['Total_floor']
test['floor_rate']=test['Current_floor']/test['Total_floor']


### Feature Engineering 11 ###

#  낙찰이 한 번 되었다고 해도 낙찰자가 돈을 못 내서 경매가 다시 일어나는 경우가 있다.
#  유찰과 다르게 이것을 재경매라고 부르는데, result라는 데이터프레임에서,
#  하나의 Auction_key당 Auction_results가 2개 이상 있을 때 재경매라고 할 수 있을 것이다.
#  이렇게 재경매가 일어난 경매를 is_re라는 변수로 구분한다.

## 이런 변수는 상상도 못했다...
result1=result.loc[result['Auction_results']=="낙찰",]

a=result1.groupby(['Auction_key'])['Auction_results'].count().reset_index()
a['Auction_results']=a['Auction_results']>1
a.columns=['Auction_key','is_re']
train=pd.merge(train,a,how='left')
test=pd.merge(test,a,how='left')


sns.factorplot('is_re','real',data=train)
#  재경매가 일어난 건일수록 target값이 높아지는 경향이 있다.


### Feature Engineering 12 ###

#  경매에서 낙찰가에 가장 중요한 영향을 미치는 요소는 시세일 것이다.
#  왜냐하면 사람들은 대부분 경매를 통해 시세차익을 노리기 때문이다.
#  그러므로 해당 아파트의 낙찰 시점(Final_auction_date)에 시세를 파악하는 것이 매우 중요한 것 같다.
#  이를 알기 위해서 외부데이터(r-one)에서 시군구별,월별 아파트의 매매가격지수 추이를 나타낸 데이터를 다운받는다.
#  모든 구는 2017년 11월의 시세를 100으로 놓고 월별로 그것의 추이를 수치화했다.
#  정확한 데이터의 출처:http://www.r-one.co.kr/rone/resis/statistics/statisticsViewer.do에서
#  '전국주택가격동향조사'에서 '월간동향'에서 '아파트'에서 '매매가격지수'에서 '매매가격지수'를 클릭한다. 여기서 서울과 부산의 데이터만 뽑아서
#  seoul_price와 busan_price라는 csv 파일에 저장한다.

## 외부데이터 수집
seoul=pd.read_csv("seoul_price_external.csv",engine="python")
busan=pd.read_csv("busan_price_external.csv",engine="python")

##  date 컬럼을 datetime으로 수정한다.
seoul['date'] = seoul['date'].astype('str').str.replace(pat="년 ",repl="-")
seoul['date'] = seoul['date'].astype('str').str.replace(pat="월",repl="-")
seoul['date'] = seoul['date']+"01"
seoul['date'] = pd.to_datetime(seoul['date'])

busan['date'] = busan['date'].astype('str').str.replace(pat="년 ",repl="-")
busan['date'] = busan['date'].astype('str').str.replace(pat="월",repl="-")
busan['date'] = busan['date']+"01"
busan['date'] = pd.to_datetime(busan['date'])


def get_expectp1(x, data):
    y = pd.DatetimeIndex(data['Appraisal_date']).year[x]
    m = pd.DatetimeIndex(data['Appraisal_date']).month[x]
    y1 = pd.DatetimeIndex(data['Final_auction_date']).year[x]
    m1 = pd.DatetimeIndex(data['Final_auction_date']).month[x]
    if (data['addr_do'].iloc[x] == "부산"):
        k = busan
    else:
        k = seoul

    k = k[['Unnamed: 0', data['addr_si'].iloc[x]]]

    result1 = k.loc[(pd.DatetimeIndex(k['Unnamed: 0']).year == y1) &
                    (pd.DatetimeIndex(k['Unnamed: 0']).month == m1), data['addr_si'].iloc[x]]

    result2 = k.loc[(pd.DatetimeIndex(k['Unnamed: 0']).year == y) &
                    (pd.DatetimeIndex(k['Unnamed: 0']).month == m), data['addr_si'].iloc[x]]
    if (len(result2) == 0):
        return np.nan
    else:
        return np.asscalar(result1) / np.asscalar(result2)



train['p_first']=train.apply(lambda x: get_expectp1(x.name,data=train),axis=1)
train['p_first']=np.where(train['p_first'].isnull(),(train['p_first']).mean(),train['p_first'])

test['p_first']=test.apply(lambda x: get_expectp1(x.name,data=test),axis=1)
test['p_first']=np.where(test['p_first'].isnull(),(test['p_first']).mean(),test['p_first'])

train['p_first_rat']=train['p_first']*train['Total_appraisal_price']/train['Minimum_sales_price']
test['p_first_rat']=test['p_first']*test['Total_appraisal_price']/test['Minimum_sales_price']

train['rat_sub_to1']=train['p_first_rat']-train['take_over_sum']/train['Minimum_sales_price']
test['rat_sub_to1']=test['p_first_rat']-test['take_over_sum']/test['Minimum_sales_price']



#  해당 아파트의 시세를 측정하는 첫 번째 방법을 소개한다. 일단 Total_appraisal_price가 Appraisal_date(감정이 일어난 날)에서 해당 아파트의 시세라고 생각할 수 있다.
#  그리고 Final_auction_date(낙찰이 일어난 날)에 시세는 해당 아파트가 속한 구가 Appraisal_date에 비해 Final_auction_date에 시세가 얼마나 많이 변했는지에 대한 비율 정보를 통해 예측할 수 있을 것이다.
#  낙찰시점 시세를 구하는 정확한 Metric: 낙찰시점 시세=((해당 구의 낙찰시점 매매가격지수)/(해당 구의 감정시점 매매가격지수))*감정가
#  이 시세에 최종적으로 Minimum_sales_price를 나누어서 p_first_rat이라는 변수를 만든다. 이렇게 Minimum_sales_price로 나누는 이유는 우리의 target은 Hammer_price가 아닌 Hammer_price/Minimum_sales_price이기 때문이다. 그러므로 시세에도 Minimum_sales_price를 나누는 것이 학습에 도움이 될 것이다.
#  rat_sub_to1는 이 p_first에 take_over_sum을 빼서 Minimum_sales_price로 나누어 준 것이다. 즉 낙찰자가 부담해야 하는 권리금액의 비율을 뺀 것이다.


### Feature Engineering 13 ###

#  ratclaim 변수: 청구금액/(감정가-최저경매가+1) ==>1을 더해주는 이유는 감정가와 최저경매가가 똑같은 경우가 있기 때문이다.
#  subclaim 변수: 감정가-최저경매가-청구금액
#  diffbetpp 변수: 청구금액-감정가
train['ratclaim']=train['Claim_price']/(train['Total_appraisal_price']-train['Minimum_sales_price']+1)
train['subclaim']=train['Total_appraisal_price']-train['Minimum_sales_price']-train['Claim_price']
train['diffbetpp']=train['Claim_price']-train['Total_appraisal_price']

test['ratclaim']=test['Claim_price']/(test['Total_appraisal_price']-test['Minimum_sales_price']+1)
test['subclaim']=test['Total_appraisal_price']-test['Minimum_sales_price']-test['Claim_price']
test['diffbetpp']=test['Claim_price']-test['Total_appraisal_price']


### Feature Engineering 14 ###

#  이제 result에 해당 Auction_key의 정보를 가지고 많은 정보를 만들어낸다. 각각의 Auction_results의 개수, 비율, 최빈값 등등을 추출해 train과 test에 merge시킨다.
## 이 과정은 무슨 의미인지 모르겠다.

from tqdm import tqdm

result_df = pd.DataFrame()
for i, df in tqdm(result.groupby('Auction_key')):
    result_df.loc[i, 'Auction_key'] = int(df['Auction_key'].values[0])
    result_df.loc[i, 'nac'] = (df['Auction_results'] == '낙찰').sum()
    result_df.loc[i, 'bae'] = (df['Auction_results'] == '배당').sum()
    result_df.loc[i, 'you'] = (df['Auction_results'] == '유찰').sum()
    result_df.loc[i, 'byun'] = (df['Auction_results'] == '변경').sum()
    result_df.loc[i, 'dae'] = (df['Auction_results'] == '대납').sum()
    result_df.loc[i, 'nacrat'] = (df['Auction_results'] == '낙찰').mean()
    result_df.loc[i, 'baerat'] = (df['Auction_results'] == '배당').mean()
    result_df.loc[i, 'yourat'] = (df['Auction_results'] == '유찰').mean()
    result_df.loc[i, 'byunrat'] = (df['Auction_results'] == '변경').mean()
    result_df.loc[i, 'daerat'] = (df['Auction_results'] == '대납').mean()
    result_df.loc[i, 'maxres'] = df['Auction_results'].value_counts().index[0]
    result_df.loc[i, 'maxnum'] = df['Auction_results'].value_counts()[0]
    result_df.loc[i, 'maxrat'] = (df['Auction_results'].value_counts()/df['Auction_results'].value_counts().sum())[0]

train = train.merge(result_df, how='left', on='Auction_key')
test = test.merge(result_df, how='left', on='Auction_key')

### Feature Engineering 15 ###

#  등기부등본(regist)에 토지별도등기가 몇 개 있는지를 toji라는 변수에 저장한다.
regist_df = pd.DataFrame()
for i, df in tqdm(regist.groupby('Auction_key')):
    regist_df.loc[i, 'Auction_key'] = int(df['Auction_key'].values[0])
    regist_df.loc[i, 'toji'] = (df['Regist_type']=="토지별도등기").sum()

train=train.merge(regist_df,how='left',on='Auction_key')
test=test.merge(regist_df,how='left',on='Auction_key')

train['toji']=np.where(train['toji'].isnull(),0,train['toji'])
test['toji']=np.where(test['toji'].isnull(),0,test['toji'])

#  토지별도등기가 있으면 사람들은 해당 부동산의 권리에 하자가 있을 위험을 감지하고 시장을 빠져나와 낙찰가가 감소한다.
sns.factorplot('toji','real',data=train)


### Feature Engineering 16 ###

## 이 변수는 여러개를 만들어도 괜찮을거 같다. 굳이 한달뿐만 아니라, 6달전 까지도 볼 필요가 있지 않을까?

#  낙찰가에 영향을 미치는 요소는 낙찰시점 시세뿐 아니라, 해당 아파트가 속한 구의 아파트 가격의 추세도 중요한 요소가 될 수 있다.
#  예를 들어 해당 구의 가격이 엄청나게 빨리 뛰고 있으면 해당 아파트를 소유하는 것이(낙찰받는 것이) 미래가치가 높을 것이라고 사람들이 판단할 것이고 자연스레 낙찰가는 상승하는 효과를 보일 것이다.
#  그래서 이 요소를 trend라는 변수로 저장한다.trend는 (낙찰시점 시세)/(낙찰시점 한달 전의 시세)로 계산한다

def get_trend(x, data):
    y = pd.DatetimeIndex(data['Final_auction_date']).year[x]
    m = pd.DatetimeIndex(data['Final_auction_date']).month[x] - 1

    if m <= 0:
        y = pd.DatetimeIndex(data['Final_auction_date']).year[x] - 1
        m = 12

    y1 = pd.DatetimeIndex(data['Final_auction_date']).year[x]
    m1 = pd.DatetimeIndex(data['Final_auction_date']).month[x]

    if data['addr_do'].iloc[x] == "부산":
        k = busan
    else:
        k = seoul

    k = k[['Unnamed: 0', data['addr_si'].iloc[x]]]

    result1 = k.loc[(pd.DatetimeIndex(k['Unnamed: 0']).year == y1) &
                    (pd.DatetimeIndex(k['Unnamed: 0']).month == m1), data['addr_si'].iloc[x]]

    result2 = k.loc[(pd.DatetimeIndex(k['Unnamed: 0']).year == y) &
                    (pd.DatetimeIndex(k['Unnamed: 0']).month == m), data['addr_si'].iloc[x]]

    return np.asscalar(result1) / np.asscalar(result2)

train['trend']=train.apply(lambda x: get_trend(x.name,data=train),axis=1)
test['trend']=test.apply(lambda x: get_trend(x.name,data=test),axis=1)



### Feature Engineering 17 ###

#  trend_first_fin은 trend와 비슷한 요소를 표현하는 또다른 변수이다.
#  trend_first_fin=(낙찰시점 시세)/(유찰이 되었을 때 그 이전 경매날짜의 시세)


def get_trend1(x, data):
    a = result.loc[(result['Auction_key'] == data['Auction_key'].iloc[x]) & (result['Auction_results'] == "유찰"),]
    a = a['Auction_date'].max()

    y = np.asscalar(pd.DatetimeIndex(pd.Series([a])).year)
    m = np.asscalar(pd.DatetimeIndex(pd.Series([a])).month)

    y1 = pd.DatetimeIndex(data['Final_auction_date']).year[x]
    m1 = pd.DatetimeIndex(data['Final_auction_date']).month[x]

    if np.asscalar(pd.Series(y).isnull()):
        y = y1
        m = m1

    if data['addr_do'].iloc[x] == "부산":
        k = busan
    else:
        k = seoul

    k = k[['Unnamed: 0', data['addr_si'].iloc[x]]]

    result1 = k.loc[(pd.DatetimeIndex(k['Unnamed: 0']).year == y1) &
                    (pd.DatetimeIndex(k['Unnamed: 0']).month == m1), data['addr_si'].iloc[x]]

    result2 = k.loc[(pd.DatetimeIndex(k['Unnamed: 0']).year == y) &
                    (pd.DatetimeIndex(k['Unnamed: 0']).month == m), data['addr_si'].iloc[x]]

    if (len(result2) == 0):
        return np.nan
    else:
        return np.asscalar(result1) / np.asscalar(result2)

train['trend_first_fin']=train.apply(lambda x: get_trend1(x.name,data=train),axis=1)
test['trend_first_fin']=test.apply(lambda x: get_trend1(x.name,data=test),axis=1)
train['trend_first_fin']=np.where(train['trend_first_fin'].isnull(),train['trend_first_fin'].mean(),train['trend_first_fin'])
test['trend_first_fin']=np.where(test['trend_first_fin'].isnull(),test['trend_first_fin'].mean(),test['trend_first_fin'])


print(train['trend_first_fin'].corr(train['real']))
plt.scatter(train['trend_first_fin'],train['real'])




### Feature Engineering 18 ###


#  해당 부동산의 시세를 좀 더 정확히 파악하기 위해 또다른 외부데이터를 사용한다.
#  이번에는 실거래가 데이터를 이용할 것이다.
#  http://rtdown.molit.go.kr/# ==> 이 사이트에 있는 실거래가 데이터(아파트별로 다 나와 있다)
#  이것을 이용해서 시세를 나타내는 또다른 변수를 만들어낼 것이다.
## ... 여기서 부터 숨쉬기 힘들어 졌다..

newdf= pd.read_csv("아파트_실거래가.csv",engine="python")
newdf=newdf.drop(['번지'],axis=1)

newdf=newdf.drop_duplicates()
a=newdf['계약일'].str.split("~",expand=True)
a.columns=['a','b']
newdf['days']=(a['a'].astype('int64')/2+a['b'].astype('int64')/2).astype('int64')
newdf['addr_bunji']=newdf['본번'].astype('str')+"-"+newdf['부번'].astype('str')
newdf=newdf.drop(['본번','부번','계약일'],axis=1)
a=newdf['시군구'].str.split(" ",expand=True)
a.columns=['a','addr_si','addr_dong','b']
a=a.drop(['a','b'],axis=1)
newdf['addr_sidong']=a['addr_si']+" "+a['addr_dong']
newdf=newdf.drop(['시군구'],axis=1)
train.loc[train['addr_bunji2'].isnull(),'addr_bunji2']=0
train.loc[~train['addr_bunji1'].isnull(),'addr_bunji']=train.loc[~train['addr_bunji1'].isnull(),'addr_bunji1'].astype('int64').astype('str')+"-"+train.loc[~train['addr_bunji1'].isnull(),'addr_bunji2'].astype('int64').astype('str')
train.loc[train['addr_bunji1'].isnull(),'addr_bunji']=np.nan

test.loc[test['addr_bunji2'].isnull(),'addr_bunji2']=0
test.loc[~test['addr_bunji1'].isnull(),'addr_bunji']=test.loc[~test['addr_bunji1'].isnull(),'addr_bunji1'].astype('int64').astype('str')+"-"+test.loc[~test['addr_bunji1'].isnull(),'addr_bunji2'].astype('int64').astype('str')
test.loc[test['addr_bunji1'].isnull(),'addr_bunji']=np.nan

newdf['year']=newdf['계약년월'].astype('str').str.slice(0,4)
newdf['month']=newdf['계약년월'].astype('str').str.slice(4,6)
newdf=newdf.drop(['계약년월'],axis=1)

train['addr_sidong']=train['addr_si']+" "+train['addr_dong']
test['addr_sidong']=test['addr_si']+" "+test['addr_dong']

newdf['tr_date']=newdf['year'].astype('str')+"-"+newdf['month'].astype('str')+"-"+newdf['days'].astype('str')
newdf=newdf.drop(['month','year','days'],axis=1)

newdf['tr_date']=newdf['tr_date'].astype('str')
newdf['tr_date']=newdf['tr_date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))

train['Final_auction_date']=train['Final_auction_date'].astype('str').str.slice(0,10).apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
train['Appraisal_date']=train['Appraisal_date'].astype('str').str.slice(0,10).apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
train['First_auction_date']=train['First_auction_date'].astype('str').str.slice(0,10).apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))

test['Final_auction_date']=test['Final_auction_date'].astype('str').str.slice(0,10).apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
test['Appraisal_date']=test['Appraisal_date'].astype('str').str.slice(0,10).apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
test['First_auction_date']=test['First_auction_date'].astype('str').str.slice(0,10).apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
newdf['거래금액(만원)']=newdf['거래금액(만원)'].astype('str').str.replace(pat=",",repl="").astype('float')
newdf.columns=['dan_name','Total_building_auction_area','price','Current_floor','built_date','road_name','addr_bunji','addr_sidong','tr_date']
a=newdf['addr_sidong'].str.split(" ",expand=True)
a.columns=['a','b']
newdf['addr_si']=a['a']
newdf['position']=newdf['addr_sidong'].astype('str')+newdf['addr_bunji'].astype('str')
train['position']=train['addr_sidong'].astype('str')+train['addr_bunji'].astype('str')
test['position']=test['addr_sidong'].astype('str')+test['addr_bunji'].astype('str')



### Feature Engineering 19 ###

#  getprice_by_bunji_finaldate의 아이디어는 해당 번지(아파트) 내에서 경매대상인 방과 면적의 차이가 일정 수준 내인 매물의 실거래가를 통해 경매대상의 시세를 파악하는 방법이다.

def getprice_by_bunji_finaldate(i, area_diff=5, data=train, method="mean"):
    a = pd.DatetimeIndex(newdf['tr_date']).year.isin([pd.DatetimeIndex(data['Final_auction_date']).year[i]])
    # a=pd.DatetimeIndex(data['Final_auction_date']).year[i]==pd.DatetimeIndex(newdf['tr_date']).year
    b = pd.DatetimeIndex(newdf['tr_date']).month.isin([pd.DatetimeIndex(data['Final_auction_date']).month[i]])

    c = newdf['position'].isin([data['position'].iloc[i]])
    # d=newdf['addr_bunji'].isin([data['addr_bunji'].iloc[i]])
    d = (data['Total_building_auction_area'].iloc[i] - newdf['Total_building_auction_area']).abs() <= area_diff
    final_cond = a & b & c & d
    if (i + 1) % 100 == 0:
        print(i + 1)
    if (method == "mean"):
        return newdf.loc[final_cond, "price"].mean()
    elif (method == "median"):
        return newdf.loc[final_cond, "price"].median()
    elif (method == "linear_regression"):
        x = pd.DataFrame(newdf.loc[(newdf['addr_si'] == train['addr_si'][i]) & (
                    pd.DatetimeIndex(data['Final_auction_date']).year[i] == pd.DatetimeIndex(
                newdf['tr_date']).year), 'Current_floor'])
        y = pd.Series(newdf.loc[(newdf['addr_si'] == train['addr_si'][i]) & (
                    pd.DatetimeIndex(data['Final_auction_date']).year[i] == pd.DatetimeIndex(
                newdf['tr_date']).year), 'price'])
        lr = LinearRegression().fit(x, y)
        return ((lr.coef_ / 3 * (train['Current_floor'].iloc[i] - newdf.loc[final_cond, "Current_floor"].median()) +
                 newdf.loc[final_cond, "price"].median()).astype('float'))

### Feature Engineering 20 ###
#  price_의 아이디어는 해당 번지(아파트) 내에서 경매대상인 방과 면적의 차이, 그리고 층 차이가 일정한 수준 내인 매물의 실거래가를 통해 경매대상의 시세를 파악하는 방법이다.

def price_(i,data=train,floor_diff=10,daydiff=100):
    a=newdf['position'].isin([data['position'].iloc[i]])
    #b=newdf['addr_sidong'].isin([data['addr_sidong'].iloc[i]])
    #a=data['addr_bunji'].iloc[i]==newdf['addr_bunji']
    #b=data['addr_sidong'].iloc[i]==newdf['addr_sidong']
    b=(data['Total_building_auction_area'].iloc[i]-newdf['Total_building_auction_area']).abs()<=5
    c=(data['Current_floor'].iloc[i]-newdf['Current_floor']).abs()<=floor_diff
    d=(newdf['tr_date']-data['Final_auction_date'].iloc[i]).abs().dt.days<=daydiff
    final_cond=a & b & c & d
    k=newdf.loc[final_cond,]
    if (i+1)%100==0:
        print(i+1)
    if(len(k)==0):
        return np.nan
    else:
        return k['price'].mean()#(k.loc[(k['tr_date']-data['Final_auction_date'].iloc[i]).abs().min()==(k['tr_date']-data['Final_auction_date'].iloc[i]).abs(),"price"]).mean()
    #return median(newdf.loc[(newdf.loc[final_cond,"tr_date"]-data['Appraisal_date']).min()==newdf[,'tr_date'],"price"])
    #if(method=="mean"):
    #return newdf.loc[(a & b & c),"price"],mean()



### Feature Engineering 21 ###

#  price_prev와 price_post는 price_와 아이디어는 같지만, 낙찰시점 전과 후의 가장 최근 몇개의 실거래가만 보는 것에서 다르다

def price_prev(i,data=train,floor_diff=10,big=3):
    a=newdf['position'].isin([data['position'].iloc[i]])
    #b=newdf['addr_sidong'].isin([data['addr_sidong'].iloc[i]])
    b=(data['Total_building_auction_area'].iloc[i]-newdf['Total_building_auction_area']).abs()<=5
    c=(data['Current_floor'].iloc[i]-newdf['Current_floor']).abs()<=floor_diff
    d=(data['Final_auction_date'].iloc[i]-newdf['tr_date']).dt.days>0
    final_cond=a & b & c & d
    k=newdf.loc[final_cond,]
    k1=(k['tr_date'])
    if (i+1)%100==0:
        print(i+1)
    return (k.loc[k1.isin(k1.nlargest(big)),"price"]).mean()


def price_post(i,data=train,floor_diff=10,small=3):
    #df[df['categories'].map(lambda x: x in string.ascii_lowercase)]
    #df[df['categories'].isin(list(string.ascii_lowercase))]
    a=newdf['position'].isin([data['position'].iloc[i]])
    #a=newdf['addr_bunji'].map(lambda x: x in data['addr_bunji'].iloc[i])
    #b=newdf['addr_sidong'].isin([data['addr_sidong'].iloc[i]])
    #b=newdf['addr_sidong'].map(lambda x: x in data['addr_sidong'].iloc[i])
    #a=data['addr_bunji'].iloc[i]==newdf['addr_bunji']
    #b=data['addr_sidong'].iloc[i]==newdf['addr_sidong']
    b=(data['Total_building_auction_area'].iloc[i]-newdf['Total_building_auction_area']).abs()<=5
    c=(data['Current_floor'].iloc[i]-newdf['Current_floor']).abs()<=floor_diff
    d=(data['Final_auction_date'].iloc[i]-newdf['tr_date']).dt.days<0
    final_cond=a & b & c & d
    k=newdf.loc[final_cond,]
    k1=(k['tr_date'])
    if (i+1)%100==0:
        print(i+1)
    return (k.loc[k1.isin(k1.nsmallest(small)),"price"]).mean()

for i in [1,5]:
    train['p_post: '+str(i)+', '+str(10)]=train.apply(lambda x: price_post(i=x.name,small=i,floor_diff=10),axis=1)
    train['p_prev: '+str(i)+', '+str(10)]=train.apply(lambda x: price_prev(i=x.name,big=i,floor_diff=10),axis=1)

for i in [5,10,50,100]:
    train['bunji with: '+str(i)]=train.apply(lambda x: getprice_by_bunji_finaldate(i=x.name,area_diff=i),axis=1)

for i in [1,5]:
    test['p_post: '+str(i)+', '+str(10)]=test.apply(lambda x: price_post(i=x.name,small=i,floor_diff=10,data=test),axis=1)
    test['p_prev: '+str(i)+', '+str(10)]=test.apply(lambda x: price_prev(i=x.name,big=i,floor_diff=10,data=test),axis=1)
for i in [5,10,50,100]:
    test['bunji with: '+str(i)]=test.apply(lambda x: getprice_by_bunji_finaldate(i=x.name,area_diff=i,data=test),axis=1)




#  이렇게 만든 변수들의 na 값은 모두 Minimum_sales_price에서 1.14를 곱한 뒤 10000을 나눈 것으로 대체한다.
#  실거래가 데이터가 실제 거래액에서 10000을 나눈 것이므로 10000으로 나누고, 1.14를 곱한 이유는 Ham_p/Min_p의 평균이 1.14 정도가 되기 때문이다.
#  그리고 나서 각 변수를 모두 Minimum_sales_price로 나눈다. 다시 한 번 더 말하지만 우리의 target은 Hammer_price가 아니라 Hammer_price/Minimum_sales_price이기 때문이다.

train['p_prev: 1, 10']=np.where(train['p_prev: 1, 10'].isnull(),train['p_post: 1, 10'],train['p_prev: 1, 10'])
#train['p_prev: 1, 10']=np.where(train['p_prev: 1, 10'].isnull(),train['bunji with: 5'],train['p_prev: 1, 10'])
train['p_prev: 1, 10']=np.where(train['p_prev: 1, 10'].isnull(),train['Minimum_sales_price']*train['p_first_rat']/10000,train['p_prev: 1, 10'])


train['p_prev: 5, 10']=np.where(train['p_prev: 5, 10'].isnull(),train['p_post: 5, 10'],train['p_prev: 5, 10'])
#train['p_prev: 5, 10']=np.where(train['p_prev: 5, 10'].isnull(),train['bunji with: 5'],train['p_prev: 5, 10'])
train['p_prev: 5, 10']=np.where(train['p_prev: 5, 10'].isnull(),train['Minimum_sales_price']*train['p_first_rat']/10000,train['p_prev: 5, 10'])


#train['p_prev: 10, 10']=np.where(train['p_prev: 10, 10'].isnull(),train['p_post: 10, 10'],train['p_prev: 10, 10'])
#train['p_prev: 10, 10']=np.where(train['p_prev: 10, 10'].isnull(),train['bunji with: 5'],train['p_prev: 10, 10'])
#train['p_prev: 10, 10']=np.where(train['p_prev: 10, 10'].isnull(),train['Minimum_sales_price']*train['p_first_rat']/10000,train['p_prev: 10, 10'])



train['p_post: 1, 10']=np.where(train['p_post: 1, 10'].isnull(),train['p_prev: 1, 10'],train['p_post: 1, 10'])
train['p_post: 5, 10']=np.where(train['p_post: 5, 10'].isnull(),train['p_prev: 5, 10'],train['p_post: 5, 10'])
#train['p_post: 10, 10']=np.where(train['p_post: 10, 10'].isnull(),train['p_prev: 10, 10'],train['p_post: 10, 10'])

test['p_prev: 1, 10']=np.where(test['p_prev: 1, 10'].isnull(),test['p_post: 1, 10'],test['p_prev: 1, 10'])
#test['p_prev: 1, 10']=np.where(test['p_prev: 1, 10'].isnull(),test['bunji with: 5'],test['p_prev: 1, 10'])
test['p_prev: 1, 10']=np.where(test['p_prev: 1, 10'].isnull(),test['Minimum_sales_price']*test['p_first_rat']/10000,test['p_prev: 1, 10'])


test['p_prev: 5, 10']=np.where(test['p_prev: 5, 10'].isnull(),test['p_post: 5, 10'],test['p_prev: 5, 10'])
#test['p_prev: 5, 10']=np.where(test['p_prev: 5, 10'].isnull(),test['bunji with: 5'],test['p_prev: 5, 10'])
test['p_prev: 5, 10']=np.where(test['p_prev: 5, 10'].isnull(),test['Minimum_sales_price']*test['p_first_rat']/10000,test['p_prev: 5, 10'])


#test['p_prev: 10, 10']=np.where(test['p_prev: 10, 10'].isnull(),test['p_post: 10, 10'],test['p_prev: 10, 10'])
#test['p_prev: 10, 10']=np.where(test['p_prev: 10, 10'].isnull(),test['bunji with: 5'],test['p_prev: 10, 10'])
#test['p_prev: 10, 10']=np.where(test['p_prev: 10, 10'].isnull(),test['Minimum_sales_price']*test['p_first_rat']/10000,test['p_prev: 10, 10'])


test['p_post: 1, 10']=np.where(test['p_post: 1, 10'].isnull(),test['p_prev: 1, 10'],test['p_post: 1, 10'])
test['p_post: 5, 10']=np.where(test['p_post: 5, 10'].isnull(),test['p_prev: 5, 10'],test['p_post: 5, 10'])
#test['p_post: 10, 10']=np.where(test['p_post: 10, 10'].isnull(),test['p_prev: 10, 10'],test['p_post: 10, 10'])7




train['bunji with: 5']=np.where(train['bunji with: 5'].isnull(),train['p_prev: 5, 10'],train['bunji with: 5'])
train['bunji with: 10']=np.where(train['bunji with: 10'].isnull(),train['p_prev: 5, 10'],train['bunji with: 10'])
train['bunji with: 50']=np.where(train['bunji with: 50'].isnull(),train['p_prev: 5, 10'],train['bunji with: 50'])
train['bunji with: 100']=np.where(train['bunji with: 100'].isnull(),train['p_prev: 5, 10'],train['bunji with: 100'])

test['bunji with: 5']=np.where(test['bunji with: 5'].isnull(),test['p_prev: 5, 10'],test['bunji with: 5'])
test['bunji with: 10']=np.where(test['bunji with: 10'].isnull(),test['p_prev: 5, 10'],test['bunji with: 10'])
test['bunji with: 50']=np.where(test['bunji with: 50'].isnull(),test['p_prev: 5, 10'],test['bunji with: 50'])
test['bunji with: 100']=np.where(test['bunji with: 100'].isnull(),test['p_prev: 5, 10'],test['bunji with: 100'])


train['p_prev: 1, 10']=train['p_prev: 1, 10']/train['Minimum_sales_price']*10000
train['p_prev: 5, 10']=train['p_prev: 5, 10']/train['Minimum_sales_price']*10000
#train['p_prev: 10, 10']=train['p_prev: 10, 10']/train['Minimum_sales_price']*10000

train['p_post: 1, 10']=train['p_post: 1, 10']/train['Minimum_sales_price']*10000
train['p_post: 5, 10']=train['p_post: 5, 10']/train['Minimum_sales_price']*10000
#train['p_post: 10, 10']=train['p_post: 10, 10']/train['Minimum_sales_price']*10000

train['bunji with: 5']=train['bunji with: 5']/train['Minimum_sales_price']*10000
train['bunji with: 10']=train['bunji with: 10']/train['Minimum_sales_price']*10000
train['bunji with: 50']=train['bunji with: 50']/train['Minimum_sales_price']*10000
train['bunji with: 100']=train['bunji with: 100']/train['Minimum_sales_price']*10000

test['p_prev: 1, 10']=test['p_prev: 1, 10']/test['Minimum_sales_price']*10000
test['p_prev: 5, 10']=test['p_prev: 5, 10']/test['Minimum_sales_price']*10000
#test['p_prev: 10, 10']=test['p_prev: 10, 10']/test['Minimum_sales_price']*10000

test['p_post: 1, 10']=test['p_post: 1, 10']/test['Minimum_sales_price']*10000
test['p_post: 5, 10']=test['p_post: 5, 10']/test['Minimum_sales_price']*10000
#test['p_post: 10, 10']=test['p_post: 10, 10']/test['Minimum_sales_price']*10000

test['bunji with: 5']=test['bunji with: 5']/test['Minimum_sales_price']*10000
test['bunji with: 10']=test['bunji with: 10']/test['Minimum_sales_price']*10000
test['bunji with: 50']=test['bunji with: 50']/test['Minimum_sales_price']*10000
test['bunji with: 100']=test['bunji with: 100']/test['Minimum_sales_price']*10000


train.to_csv("train_ready.csv",index=False)
test.to_csv("test_ready.csv",index=False)

train = train.drop(['addr_bunji', 'addr_sidong', 'position'], axis=1)
test = test.drop(['addr_bunji', 'addr_sidong', 'position'], axis=1)

train = train.drop(
    ['road_name', 'addr_etc', 'First_auction_date', 'Appraisal_date', 'addr_si', 'addr_dong', 'addr_li', 'addr_san'],
    axis=1)
test = test.drop(
    ['road_name', 'addr_etc', 'First_auction_date', 'Appraisal_date', 'addr_si', 'addr_dong', 'addr_li', 'addr_san'],
    axis=1)

train = train.drop(['Creditor', 'Specific', 'Appraisal_company', 'Close_result', 'Final_result'], axis=1)
test = test.drop(['Creditor', 'Specific', 'Appraisal_company', 'Close_result', 'Final_result'], axis=1)

from datetime import datetime

today = datetime.today()

train['diff_today_final_date'] = (pd.to_datetime(train['Final_auction_date']) - today).dt.days
# test['diff_today_appraisal_date'] = (pd.to_datetime(df_test['Appraisal_date']) - today).astype(int)
train = train.drop(['Final_auction_date', 'Close_date', 'Preserve_regist_date'], axis=1)

test['diff_today_final_date'] = (pd.to_datetime(test['Final_auction_date']) - today).dt.days
test = test.drop(['Final_auction_date', 'Close_date', 'Preserve_regist_date'], axis=1)

le = LabelEncoder()
le_count = 0

for col in train:
    if train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(train[col].unique())) <= 2:
            # Train on the training data
            le.fit(train[col])
            # Transform both training and testing data
            train[col] = le.transform(train[col])

            # Keep track of how many columns were label encoded
            le_count += 1
print(le_count)

train = pd.get_dummies(train)

le = LabelEncoder()
le_count = 0

for col in test:
    if test[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(test[col].unique())) <= 2:
            # Train on the training data
            le.fit(test[col])
            # Transform both training and testing data
            test[col] = le.transform(test[col])

            # Keep track of how many columns were label encoded
            le_count += 1
print(le_count)

test = pd.get_dummies(test)
train = train.drop(['addr_bunji1', 'addr_bunji2', 'road_bunji1', 'road_bunji2'], axis=1)
test = test.drop(['addr_bunji1', 'addr_bunji2', 'road_bunji1', 'road_bunji2'], axis=1)
