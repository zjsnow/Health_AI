'''
    5-fold 3-model 2-layer stacking

    @author abaisa
    2017 05 07
'''
import numpy as np
import pandas as pd
import xgboost as xgb
from common.my_fun import error_fun

from model.lightGBM_model import lightBGM_model
from model.linear_model import linear_model


# 增加模型在这里配置实现
# 这里所有生成对model函数均保证  输入X和Y 返回一个训练好的model
def get_model(model_number):
    if model_number == 0:
        return lightBGM_model
    if model_number == 1:
        return linear_model

# return_part 是从0开始的
def split_data(X, return_part, part_cnt):
    data_count =X.shape[0]
    start_row = (data_count // part_cnt) * return_part
    finish_row = (data_count // part_cnt) * (return_part + 1)
    if return_part == part_cnt - 1: finish_row = data_count
    if type(X) == pd.DataFrame:
        return pd.concat([X.iloc[:start_row, :], X.iloc[finish_row:, :]]), X.iloc[start_row:finish_row, :]
    else:
        return pd.concat([X.iloc[:start_row], X.iloc[finish_row:]]), X.iloc[start_row:finish_row]

# 特征选择的代码没有整合
# 可以设置一个阈值，当模型低于这个阈值当时候重新跑
def stacking_layer_1(train_X, train_Y, test_X):
    layer_2_trainX, layer_2_testX = [], []
    for model_num in range(1):
        train_predict_list, test_predict_list = [], []
        model_untrained = get_model(model_num)
        # 5-fold cross verify
        for i in range(5):
            X1, X2 = split_data(train_X, i, 5)
            Y1, Y2 = split_data(train_Y, i, 5)
            Y1 = np.log1p(Y1)
            model = model_untrained(X1, Y1)
            P = model.predict(X2)
            P = np.expm1(P)
            TP = np.expm1(model.predict(test_X))
            error = error_fun(P, Y2)[1]
            print('layer 1 model ' + str(model_num) + ' fold ' + str(i) + ' error >> ' + str(error))

            # 保留预测结果P，作为第二层模型训练输入
            train_predict_list.append(pd.Series(P))
            # 对test进行预测并保留，使用训练后第二层模型对其进行预测
            test_predict_list.append(pd.Series(TP))

        train_predict = pd.concat(train_predict_list, axis=0, ignore_index=True)

        test_predict = test_predict_list[0]
        for j in range(1, len(test_predict_list)):
            test_predict = test_predict + test_predict_list[j]
        test_predict = test_predict / 5

        error = error_fun(train_predict, train_Y.values)
        print('layer 1 model ' + str(model_num) + ' final error >> ' + str(error))

        layer_2_trainX.append(train_predict)
        layer_2_testX.append(test_predict)

    # 将list转为DataFrame
    layer_2_trainX = pd.DataFrame(layer_2_trainX).T
    layer_2_testX = pd.DataFrame(layer_2_testX).T

    print('layer_2_trainX shape : ' + str(layer_2_trainX.shape))
    print('layer_2_testX shape : ' + str(layer_2_testX.shape))

    return layer_2_trainX, layer_2_testX

def stacking_layer_2(train_X, train_Y, test_X):
    # 训练参数设置和执行
    params = {
        'n_estimators' : 240,
        'max_depth' : 6,
        'min_child_weight' : 1,
        'colsample_bytree' : 0.6,
        'subsample' : 0.9}

    xg_train = xgb.DMatrix(train_X, label = train_Y)
    xgboost_model = xgb.train(params, xg_train)

    xg_test = xgb.DMatrix(test_X)
    xg_res = xgboost_model.predict(xg_test)

    return xg_res

def stacking(train_X, train_Y, test_X):
    train_X1, test_X1 = stacking_layer_1(train_X, train_Y, test_X)
    res = stacking_layer_2(train_X1, train_Y, test_X1)
    return res

if __name__ == '__main__':
    from common.data_fun import get_train_X, get_train_Y
    from sklearn.model_selection import train_test_split

    x1, x2, y1, y2 = train_test_split(get_train_X(), get_train_Y(), test_size=0.2)

    predict = stacking(x1, y1, x2)
    error = error_fun(predict, y2)[1]

    print('stacking error: ' + str(error))