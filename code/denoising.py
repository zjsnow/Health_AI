import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei']
'''
训练集中的特征值离群点样本进行了去除
'''
def denoising():
    # # 根据与y的线性关系的强弱画热点图,只画前20个
    # def plot_heatmap(train_data, y_col):
    #     corr = np.abs(train_data.corr())  # 计算皮尔森相关系数，放回Dataframe类型
    #     k = 20
    #     cols = corr.nlargest(k, y_col)[y_col].index  # nlargest()按照y选取最大的前20行，取得y列，再得到与y最相关的20个属性
    #     print(list(cols))
    #     max_corr = train_data[cols].corr()
    #     sns.heatmap(max_corr, cmap='Blues', square=True, annot=True)  # square=True表示每个特征都是正方形
    #
    # # 数值特征画线性回归图，观察数值特征和售价是否有线性关系，若没有明显的线性关系，转为分类特征处理
    # def plot_num_features(train_data, y_label, num_features_list):
    #     def my_regplot(x, y, **kwargs):
    #         sns.regplot(x=x, y=y)  # 画出散点图并进行线性拟合，若设order=2则进行二次曲线拟合
    #
    #     df = pd.melt(train_data, id_vars=[y_label], value_vars=num_features_list, var_name='features',
    #                  value_name='value')
    #     g = sns.FacetGrid(df, col='features', col_wrap=3, sharex=False, sharey=False)
    #     g.map(my_regplot, "value", y_label)

    # 画图了解数值特征的分布，去除离群点
    # y1_train=pd.concat([y_train.iloc[:,1],x_train],axis=1)
    # plot_heatmap(y1_train,'收缩压')
    # plot_num_features(y1_train,'收缩压',['2410', '30006', '1850', '0105', '2168', 'A703', '1345', '1115','1112', '269012', '313',  '100012'])
    # y2_train=pd.concat([y_train.iloc[:,2],x_train],axis=1)
    # plot_heatmap(y2_train,'舒张压')
    # plot_num_features(y2_train,'舒张压',['269012', '313', '2413', '809049','809050', '2410', '979010', 'A703', '709001', '2412', '191', '269013', '31'])
    # plt.show()
    # y3_train=pd.concat([y_train.iloc[:,3],x_train],axis=1)
    # # # plot_heatmap(y3_train,'血清甘油三酯')
    # plot_num_features(y3_train,'血清甘油三酯',['819018', '979010', '1844', '2410', '1107', '279001', '2413'])
    # # plt.show()
    # y4_train=pd.concat([y_train.iloc[:,4],x_train],axis=1)
    # # plot_heatmap(y4_train,'血清高密度脂蛋白')
    # plot_num_features(y4_train,'血清高密度脂蛋白',['819014', '191', '809012', '279003', '2406', '313', '2168', '1844', '819009'])
    # plt.show()
    # y5_train=pd.concat([y_train.iloc[:,5],x_train],axis=1)
    # # plot_heatmap(y5_train,'血清低密度脂蛋白')
    # # plot_num_features(y5_train,'血清低密度脂蛋白',['191', '809050', '2168', '100012', '809001', '459154', '321', '269013', '979017'])
    # plt.show()

    train_data=pd.read_csv('../data/train_data_num_ch.csv',encoding='utf-8')
    x_train=train_data.iloc[:,6:].astype(float)
    y_train=train_data.iloc[:,:6]
    #通过画图，去除重要特征中的噪点
    drop_index=[]
    drop_index.append(x_train.sort_values(by = '2410',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '1850',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = 'A703',ascending = True)[:2].index)
    drop_index.append(x_train.sort_values(by = '1115',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '1112',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '100012',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '709001',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '2412',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '1117',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '819020',ascending = True)[:1].index)
    drop_index.append(x_train.sort_values(by = '1107',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '1844',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '819018',ascending = True)[:1].index)
    drop_index.append(x_train.sort_values(by = '2406',ascending = True)[:2].index)
    drop_index.append(x_train.sort_values(by = '819009',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '819014',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '1107',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '100008',ascending = False)[:1].index)

    #lightgbm中重要特征的离群点
    drop_index.append(x_train.sort_values(by = '2405',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '1850',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '1115',ascending = False)[:2].index)
    drop_index.append(x_train.sort_values(by = '1815',ascending = False)[:1].index)
    drop_index.append(x_train.sort_values(by = '2404',ascending = True)[:3].index)
    drop_index.append(x_train.sort_values(by = '1814',ascending = False)[:2].index)
    drop_index.append(x_train.sort_values(by = '2333',ascending = False)[:2].index)
    drop_index.append(x_train.sort_values(by = '10004',ascending = True)[:1].index)
    drop_index.append(x_train.sort_values(by = '192',ascending = False)[:2].index)
    drop_list=[]
    for i in drop_index:
        drop_list=drop_list+list(i)
    for i in set(drop_list):
        x_train.drop(i,inplace=True)
    print('去除特征值离群的样本数：%d'%len(set(drop_list))) #去除了28个特征值离群的样本
    train_data=pd.concat([y_train,x_train],axis=1,join='inner')
    train_data.to_csv('../data/train_data_num_ch_last.csv',index=False,encoding='utf-8')

if __name__ == '__main__':
    denoising()