# coding: utf-8

import pandas as pd
import numpy as np
import datetime
from constants import constants as x

df_new = pd.read_csv('jun_before_input_df.csv').iloc[:,1:]

def num2datetime(x):
    if type(x) == str:
        return datetime.datetime.strptime(x, '%Y%m%d')
    else:
        return datetime.datetime.strptime(str(int(x)), '%Y%m%d')
    
def get_age(after, before):
    try:
        if after is None or before is None:
            return -1
        
        return int((after - before) / 10000)
    except Exception as ex:
        print(ex)
        return -1
    
    
    
window = 5
#split_date = '2017-05-01'
tables = []
train_data = []
clm = []
buy_from, buy_till = 2*7, 6*7
print(int(buy_from/7), "W 以降", int(buy_till/7), "W 以内に来店があるか")
for cid, tbl in df_new.groupby('member_id'):
    sales_date = tbl.sales_date.values
    # days_delta = -1 というのは、はじめて来店した日を意味する
    d = [-1] + [( num2datetime(sales_date[i]) - num2datetime((sales_date[i - 1])) ).days for i in range(1, len(sales_date))]
    tbl_tmp = tbl.assign(days_delta = d)
    for i in range(tbl_tmp.shape[0] - 1): # 購買判定(bought)ができるには tbl_tmp.shape[0] - 1でないといけない！
        d_tmp = [cid]
        for w in range(consts.n_window):
            if i - w >= 0:
                d_tmp += [tbl_tmp.iloc[i - w].days_delta,
                          get_age(tbl_tmp.iloc[i - w].sales_date, tbl_tmp.iloc[i - w].birthday),
                          tbl_tmp.iloc[i - w].shop_id,
                          tbl_tmp.iloc[i - w].item_num
                         ]
            else:
                d_tmp += [-1, -1, '-', -1]
        d_tmp += [str(tbl_tmp.iloc[0].gender), str(int(tbl_tmp.iloc[0].pref_cd))]

        shops = tbl_tmp.iloc[i:].shop_id.values # 店舗判定
        ss = (shops == shops[0]).astype(int) # i = 0 は今日（過去データ）, i > 0 は未来（予測データ）
        
        days = tbl_tmp.iloc[i:].days_delta.values # 日付判定
        ds = (buy_from <= np.cumsum(days)) * (buy_till >= np.cumsum(days)) # 来店範囲, buy_from > 0なら"今日"を除ける
        ds_int = ds.astype(int)
        both = ss * ds_int
        both = both[1:]

        
        if np.sum(both) > 0:
            bought_judge = 1
        elif np.sum(both) == 0:
            bought_judge = 0
        else:
            print("Error occured!")
            
        d_tmp.append(bought_judge)
        train_data.append(d_tmp)
        
clm  = ['member_id']
for w in range(1, consts.n_window + 1):
    clm += ['days_delta_{}'.format(w), 'age_{}'.format(w), 'shop_id_{}'.format(w), 'item_num_{}'.format(w)]
clm += ['gender', 'pref_cd', 'bought']

df_train_data = pd.DataFrame(train_data, columns=clm)
df_train_data.to_csv('jun_train_data{}_2w6w.csv'.format(consts.n_window))