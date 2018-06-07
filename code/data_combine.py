import pandas as pd

'''
将数值特征和中文特征整合,并去除不合理的y值
'''
def data_combine():
    train_data_num=pd.read_csv('../data/train_data_num_final.csv',encoding='utf-8')
    test_data_num=pd.read_csv('../data/test_data_num_final.csv',encoding='utf-8')
    train_data_ch=pd.read_csv('../data/train_data_ch_final.csv',encoding='utf-8')
    test_data_ch=pd.read_csv('../data/test_data_ch_final.csv',encoding='utf-8')

    x_train_num=train_data_num.ix[:,6:]
    x_train_ch=train_data_ch.ix[:,6:]
    y_train=train_data_ch.ix[:,:6]

    x_test_num=test_data_num.ix[:,1:]
    x_test_ch=test_data_ch.ix[:,1:]
    id_test=test_data_num.ix[:,0]

    train_data=pd.concat([y_train,x_train_num,x_train_ch],axis=1)
    test_data=pd.concat([id_test,x_test_num,x_test_ch],axis=1)
    print("训练集列数：%d" %len(train_data.columns))
    print("测试集列数：%d" %len(test_data.columns))

    drop_forever_id=['fa04c8db6d201b9f705a00c3086481b0',  #错误样本的id
              '7685d48685028a006c84070f68854ce1',
              '798d859a63044a8a5addf1f8c528629e',
              'd9919661f0a45fbcacc4aa2c1119c3d2',
              'de82a4130c4907cff4bfb96736674bbc',
              'bd0322cf42fc6c2932be451e0b54ed02']
    train_data.set_index(["vid"], inplace=True)
    train_data.drop(drop_forever_id,inplace=True)
    train_data.to_csv('../data/train_data_num_ch.csv',encoding='utf-8')
    test_data.to_csv('../data/test_data_num_ch.csv', index=False,encoding='utf-8')

if __name__ == '__main__':
    data_combine()