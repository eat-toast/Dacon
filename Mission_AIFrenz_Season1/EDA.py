import pandas as pd
import os

os.chdir(os.path.join(os.getcwd(), 'Mission_AIFrenz_Season1'))# 디렉토리 변경
data_path =  os.path.join(os.getcwd(), 'data')

train = pd.read_csv(os.path.join(data_path, 'train.csv'))

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