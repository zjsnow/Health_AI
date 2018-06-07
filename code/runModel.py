'''
@author = the_ares

第一次：
平均指标一：  0.013651269857557722
平均指标二：  0.017395353604368318
平均指标三：  0.06998087422854625
平均指标四：  0.01115109466518095
平均指标五：  0.03084397104953445
total error: 0.028604512681037537
第二次：
平均指标一：  0.013641862040361344
平均指标二：  0.01769532835977438
平均指标三：  0.0706033652458806
平均指标四：  0.011133964094338634
平均指标五：  0.031195402975209598
total error: 0.028853984543112908

第三次：
平均指标一：  0.013581889032525815
平均指标二：  0.017557552426124095
平均指标三：  0.07158693182594762
平均指标四：  0.011057074042298397
平均指标五：  0.03143580222502992
total error: 0.029043849910385174

第四次：
平均指标一：  0.013689294581867207
平均指标二：  0.017544034517411263
平均指标三：  0.07084274395628537
平均指标四：  0.010880196351605547
平均指标五：  0.03110628155237457
total error: 0.02881251019190879
第五次：
平均指标一：  0.013741637361167115
平均指标二：  0.017346806860714416
平均指标三：  0.07026405280661639
平均指标四：  0.010988694901117747
平均指标五：  0.03146738939347504
total error: 0.028761716264618143
'''
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import gc
import time

train_path = '../data/train_data_num_ch_last.csv'
test_path = '../data/test_data_num_ch.csv'


def change_types():
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    collumns = [col for col in train.columns if col !='vid']
    train_data = train[collumns].astype(float)
    train = pd.concat([train['vid'],train_data],axis = 1)

    train.to_csv('../data/train.csv',index = False,encoding = 'utf-8')
    features = [feature for feature in collumns if feature not in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']]
    test_data = test[features].astype(float)
    test = pd.concat([test['vid'],test_data],axis = 1)
    test.to_csv('../data/test.csv',index = False,encoding = 'utf-8')


def save_model(preds):
    averge_preds_1 = preds[0]
    averge_preds_2 = preds[1]
    averge_preds_3 = preds[2]
    averge_preds_4 = preds[3]
    averge_preds_5 = preds[4]

    test_ = pd.read_csv(test_path)
    vid_test=np.array(test_['vid'])
    print(type(vid_test))
    vid_test.shape=(vid_test.shape[0],1)
    averge_preds_1.shape=(averge_preds_1.shape[0],1)
    averge_preds_2.shape=(averge_preds_2.shape[0],1)
    averge_preds_3.shape=(averge_preds_3.shape[0],1)
    averge_preds_4.shape=(averge_preds_4.shape[0],1)
    averge_preds_5.shape=(averge_preds_5.shape[0],1)
    result=np.hstack((vid_test,averge_preds_1,averge_preds_2,averge_preds_3,averge_preds_4,averge_preds_5))
    Time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    np.savetxt('../submit/submit_' + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.csv',result,fmt='%s',delimiter=',')


def create_model_and_predict():
    preds = []
    train_ = pd.read_csv('../data/train.csv')
    test_ = pd.read_csv('../data/test.csv')

    gg = [c for c in train_.columns if c not in ['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']]  # 102列
    test_data = test_[gg]
    features = [feature for feature in gg if feature != 'vid']

    train_X = train_[features]
    test_X = test_data[features]

    def error_(pret, target):
        a = np.log(pret + 1)
        b = np.log(target + 1)
        c = b - a
        d = pow(c, 2)
        e = sum(d) / len(a)
        error = e
        return 'error', float(error)

    ##########################################
    val_sum_error_1 = 0
    preds_sum_1_log = 0

    for i in range(5):
        train_y_1 = train_['收缩压']
        x1, x2, y1, y2 = train_test_split(train_X, train_y_1, test_size=0.2)
        y1_log = np.log(y1)

        xgb_model_1 = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            min_child_weight=4,
            colsample_bytree=0.7,
            subsample=0.7)
        lgb_model_1 = lgb.LGBMRegressor(learning_rate=0.07, num_leaves=41, n_estimators=110, random_state=0)

        xgb_model_1.fit(x1, y1_log, verbose=True)
        lgb_model_1.fit(x1, y1_log, verbose=True)

        xgb_val_1_log = xgb_model_1.predict(x2)
        lgb_val_1_log = lgb_model_1.predict(x2)
        val_1_log = 0.6 * xgb_val_1_log + 0.4 * lgb_val_1_log
        val_1 = np.exp(val_1_log)
        print(error_(val_1, y2))
        val_1_error = error_(val_1, y2)

        xgb_preds_1_log = xgb_model_1.predict(test_X)
        lgb_preds_1_log = lgb_model_1.predict(test_X)
        preds_1_log = 0.6 * xgb_preds_1_log + 0.4 * lgb_preds_1_log

        preds_sum_1_log += preds_1_log
        val_sum_error_1 += val_1_error[1]

        del x1, x2, y1, y2
        gc.collect()
    averge_error_1 = val_sum_error_1 / 5

    averge_preds_1_log = preds_sum_1_log / 5
    averge_preds_1 = np.exp(averge_preds_1_log)
    print(averge_preds_1)
    preds.append(averge_preds_1)
    #######################################
    train_y_2 = train_['舒张压']
    val_sum_error_2 = 0
    preds_sum_2_log = 0

    for j in range(5):
        x1, x2, y1, y2 = train_test_split(train_X, train_y_2, test_size=0.2)
        y1_log = np.log(y1)
        xgb_model_2 = xgb.XGBRegressor(
            n_estimators=180,
            max_depth=5,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=1)

        lgb_model_2 = lgb.LGBMRegressor(num_leaves=36, n_estimators=140, random_state=0, learning_rate=0.06)

        xgb_model_2.fit(x1, y1_log, verbose=True)
        lgb_model_2.fit(x1, y1_log, verbose=True)

        xgb_val_2_log = xgb_model_2.predict(x2)
        lgb_val_2_log = lgb_model_2.predict(x2)
        val_2_log = 0.6 * xgb_val_2_log + 0.4 * lgb_val_2_log
        val_2 = np.exp(val_2_log)
        print(error_(val_2, y2))
        val_2_error = error_(val_2, y2)

        xgb_preds_2_log = xgb_model_2.predict(test_X)
        lgb_preds_2_log = lgb_model_2.predict(test_X)
        preds_2_log = 0.6 * xgb_preds_2_log + 0.4 * lgb_preds_2_log

        preds_sum_2_log += preds_2_log
        val_sum_error_2 += val_2_error[1]

        del x1, x2, y1, y2
        gc.collect()

    averge_error_2 = val_sum_error_2 / 5

    averge_preds_2_log = preds_sum_2_log / 5
    averge_preds_2 = np.exp(averge_preds_2_log)
    print(averge_preds_2)

    preds.append(averge_preds_2)
    ########################################
    train_y_3 = train_['血清甘油三酯']
    val_sum_error_3 = 0
    preds_sum_3_log = 0
    for q in range(5):
        x1, x2, y1, y2 = train_test_split(train_X, train_y_3, test_size=0.2)
        y1_log = np.log(y1)
        xgb_model_3 = xgb.XGBRegressor(n_estimators=240, max_depth=6, min_child_weight=1, colsample_bytree=0.6,
                                       subsample=0.9)
        lgb_model_3 = lgb.LGBMRegressor(num_leaves=36, n_estimators=100, learning_rate=0.07, random_state=0)

        xgb_model_3.fit(x1, y1_log, verbose=True)
        lgb_model_3.fit(x1, y1_log, verbose=True)

        xgb_val_3_log = xgb_model_3.predict(x2)
        lgb_val_3_log = lgb_model_3.predict(x2)
        val_3_log = 0.6 * xgb_val_3_log + 0.4 * lgb_val_3_log
        val_3 = np.exp(val_3_log)
        print(error_(val_3, y2))
        val_3_error = error_(val_3, y2)

        xgb_preds_3_log = xgb_model_3.predict(test_X)
        lgb_preds_3_log = lgb_model_3.predict(test_X)
        preds_3_log = 0.6 * xgb_preds_3_log + 0.4 * lgb_preds_3_log

        preds_sum_3_log += preds_3_log
        val_sum_error_3 += val_3_error[1]

        del x1, x2, y1, y2
        gc.collect()
    averge_error_3 = val_sum_error_3 / 5

    averge_preds_3_log = preds_sum_3_log / 5
    averge_preds_3 = np.exp(averge_preds_3_log)
    print(averge_preds_3)
    preds.append(averge_preds_3)
    ##########################################
    train_y_4 = train_['血清高密度脂蛋白']
    val_sum_error_4 = 0
    preds_sum_4_log = 0

    for m in range(5):
        x1, x2, y1, y2 = train_test_split(train_X, train_y_4, test_size=0.2)
        y1_log = np.log(y1)
        xgb_model_4 = xgb.XGBRegressor(n_estimators=100, max_depth=9, min_child_weight=6, colsample_bytree=0.9,
                                       subsample=0.6)
        lgb_model_4 = lgb.LGBMRegressor(n_estimators=100, num_leaves=101, random_state=0, learning_rate=0.1)

        xgb_model_4.fit(x1, y1_log, verbose=True)
        lgb_model_4.fit(x1, y1_log, verbose=True)

        xgb_val_4_log = xgb_model_4.predict(x2)
        lgb_val_4_log = lgb_model_4.predict(x2)
        val_4_log = 0.6 * xgb_val_4_log + 0.4 * lgb_val_4_log
        val_4 = np.exp(val_4_log)
        print(error_(val_4, y2))
        val_4_error = error_(val_4, y2)

        xgb_preds_4_log = xgb_model_4.predict(test_X)
        lgb_preds_4_log = lgb_model_4.predict(test_X)
        preds_4_log = 0.6 * xgb_preds_4_log + 0.4 * lgb_preds_4_log

        preds_sum_4_log += preds_4_log
        val_sum_error_4 += val_4_error[1]

        del x1, x2, y1, y2
        gc.collect()
    averge_error_4 = val_sum_error_4 / 5

    averge_preds_4_log = preds_sum_4_log / 5
    averge_preds_4 = np.exp(averge_preds_4_log)
    print(averge_preds_4)
    preds.append(averge_preds_4)
    ############################################
    train_y_5 = train_['血清低密度脂蛋白']
    val_sum_error_5 = 0
    preds_sum_5_log = 0
    for n in range(5):
        x1, x2, y1, y2 = train_test_split(train_X, train_y_5, test_size=0.2)
        y1_log = np.log(y1)
        xgb_model_5 = xgb.XGBRegressor(n_estimators=200, max_depth=9, min_child_weight=2, colsample_bytree=0.9,
                                       subsample=0.8)
        lgb_model_5 = lgb.LGBMRegressor(learning_rate=0.06, num_leaves=121, n_estimators=100, random_state=0)

        xgb_model_5.fit(x1, y1_log, verbose=True)
        lgb_model_5.fit(x1, y1_log, verbose=True)

        xgb_val_5_log = xgb_model_5.predict(x2)
        lgb_val_5_log = lgb_model_5.predict(x2)
        val_5_log = 0.6 * xgb_val_5_log + 0.4 * lgb_val_5_log
        val_5 = np.exp(val_5_log)
        print(error_(val_5, y2))
        val_5_error = error_(val_5, y2)

        xgb_preds_5_log = xgb_model_5.predict(test_X)
        lgb_preds_5_log = lgb_model_5.predict(test_X)
        preds_5_log = 0.6 * xgb_preds_5_log + 0.4 * lgb_preds_5_log

        preds_sum_5_log += preds_5_log
        val_sum_error_5 += val_5_error[1]

        del x1, x2, y1, y2
        gc.collect()
    averge_error_5 = val_sum_error_5 / 5

    averge_preds_5_log = preds_sum_5_log / 5
    averge_preds_5 = np.exp(averge_preds_5_log)
    print(averge_preds_5)
    preds.append(averge_preds_5)
    ###################################################
    print('平均指标一： ', averge_error_1)
    print('平均指标二： ', averge_error_2)
    print('平均指标三： ', averge_error_3)
    print('平均指标四： ', averge_error_4)
    print('平均指标五： ', averge_error_5)
    print('total error:', (averge_error_1 + averge_error_2 + averge_error_3 + averge_error_4 + averge_error_5) / 5)
    ################################################

    return preds
