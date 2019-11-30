import pandas as pd
import os

os.getcwd()
# 데이터가 들어있는 경로
external_path = os.path.join(os.getcwd(), 'dacon_external_data')

# 실거래가 리스트
data_path_list = [x for x in os.listdir(external_path) if '실거래가' in x ]

# 15번째 컬럼 정보가 들어있음
# 16번째부터 실제 데이터
data = pd.DataFrame()
for i in range(len(data_path_list)):

    path = os.path.join( external_path, data_path_list[i] )

    temp_data = pd.read_excel(path)
    col = temp_data.iloc[15].values
    temp_data = temp_data.iloc[16:]
    temp_data.columns = col

    data = pd.concat([data, temp_data])

    print(i, ' / ', len(data_path_list) )

data = data.drop_duplicates()  # (1113342, 12)

file_name = '아파트_실거래가.csv'

save_path = os.path.join(external_path, file_name)
data.to_csv(save_path , index = False)