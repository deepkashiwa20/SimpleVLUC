import csv
import pandas as pd
import numpy as np
import os
import shutil
import sys
import time
from datetime import datetime
import Metrics
from Param import *

def getXSYS(data):
    XS, YS = [], []  
    for i in range(data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
        x = data[i:i+TIMESTEP_IN, :, :, :]
        y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS

def HistoricalAverage(data):
    all_days = [i.strftime('%Y-%m-%d') for i in pd.date_range(START_DATE, END_DATE, freq='1d')]
    his_days = ['2011-03-01', '2011-03-02', '2011-03-03', '2011-03-04', '2011-03-07', '2011-03-08', '2011-03-09', '2011-03-10']
    his_data = []
    for day in his_days:
        index = all_days.index(day)
        his_data.append(data[index*DAY_TIMESTAMP:(index+1)*DAY_TIMESTAMP, :, :, :])
    his_data = np.array(his_data)
    print(his_data.shape)
    his_avg = np.mean(his_data, axis=0)
    XS_avg, YS_avg = getXSYS(his_avg)
    return XS_avg, YS_avg
    
def testModel(name, mode, all_data, test_data):
    _, YS = getXSYS(test_data)
    _, YS_pred = HistoricalAverage(all_data)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (name, mode, MSE, RMSE, MAE, MAPE))

################# Parameter Setting #######################
MODELNAME = 'HistoricalAverage'
KEYWORD = f'Density_{DATANAME}_{MODELNAME}' + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../save/' + KEYWORD
################# Parameter Setting #######################

def main():   
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)
    
    data = np.load(DATAPATH)
    test_data = data[TRAIN_DAYS*DAY_TIMESTAMP:, :, :, :]
    print(data.shape, test_data.shape)
        
    print(KEYWORD, 'testing started', time.ctime())
    testModel(MODELNAME, 'TEST', data, test_data)
    print(KEYWORD, 'testing ended', time.ctime())

if __name__ == '__main__':
    main()