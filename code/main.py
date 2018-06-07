from data_cleaning import data_cleaning
from data_cleaning_num import data_cleaning_num
from data_cleaning_ch import data_cleaning_ch
from data_combine import data_combine
from denoising import denoising

from runModel import change_types
from runModel import create_model_and_predict
from runModel import save_model

if __name__=="__main__":
    # data pre process
    print('data_cleaning start')
    data_cleaning()
    print('\ndata_cleaning_num start')
    data_cleaning_num()
    print('\ndata_cleaning_ch start')
    data_cleaning_ch()
    print('\ndata_combine start')
    data_combine()
    print('\ndenoising start')
    denoising()

    # fit and predict
    print('\nchange types ....')
    change_types()
    print('\ncreate model......')
    preds = create_model_and_predict()
    save_model(preds)
    print('\nsave result done!!')
