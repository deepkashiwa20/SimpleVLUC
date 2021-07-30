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

def CopyLastWeek(XS, YS):
    return XS

def testModel(name, mode, XS, YS):
    YS_pred = CopyLastWeek(XS, YS)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (name, mode, MSE, RMSE, MAE, MAPE))

def trainModel(name, mode, XS, YS):
    YS_pred = CopyLastWeek(XS, YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (name, mode, MSE, RMSE, MAE, MAPE))

################# Parameter Setting #######################
MODELNAME = 'CopyLastFrame'
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
    train_data = data[:TRAIN_DAYS*DAY_TIMESTAMP, :, :, :]
    test_data = data[TRAIN_DAYS*DAY_TIMESTAMP:, :, :, :]
    print(data.shape, train_data.shape, test_data.shape)
        
    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(train_data)
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'TRAIN', trainXS, trainYS)
    print(KEYWORD, 'training ended', time.ctime())
    
    print()
    
    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS(test_data)
    print('TEST XS.shape YS,shape', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'TEST', testXS, testYS)
    print(KEYWORD, 'testing ended', time.ctime())

if __name__ == '__main__':
    main()