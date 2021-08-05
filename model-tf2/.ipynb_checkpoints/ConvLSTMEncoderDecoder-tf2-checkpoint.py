import csv
import numpy as np
import os
import shutil
import sys
import time
from datetime import datetime
import random
import tensorflow as tf
from keras import backend as K
from keras.models import load_model, Model, Sequential
from keras.layers import Input, TimeDistributed, Flatten, RepeatVector, Reshape, concatenate, add, Dropout, Embedding, Lambda
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler
from Param import *
import Metrics

def getXSYS(data):
    XS, YS = [], []  
    for i in range(data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
        x = data[i:i+TIMESTEP_IN, :, :, :]
        y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS

def getModel():
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3,3), input_shape=(None, HEIGHT, WIDTH, CHANNEL), padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same', return_sequences=False))
    seq.add(BatchNormalization())
    seq.add(Lambda(lambda x: K.concatenate([x[:, np.newaxis, :, :, :]] * TIMESTEP_OUT, axis=1)))
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=CHANNEL, kernel_size=(3, 3), padding='same', return_sequences=True, activation='relu'))
    return seq

def testModel(name, mode, XS, YS):
    print('Model Evaluation Started ...', time.ctime())
    assert os.path.exists(os.path.join(PATH, f'{name}.h5')), 'model is not existing'
    model = getModel()
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.load_weights(os.path.join(PATH, f'{name}.h5'))
    model.summary()
    
    YS_pred = model.predict(XS)
    print(XS.shape, YS.shape, YS_pred.shape)
    YS, YS_pred = YS * MAX_VALUE, YS_pred * MAX_VALUE
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(os.path.join(PATH, f'{name}_prediction_scores.txt'), 'a') as f:
        f.write("all steps, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print("all steps, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (name, mode, MSE, RMSE, MAE, MAPE))
    
    with open(os.path.join(PATH, f'{name}_prediction_scores.txt'), 'a') as f:
        for i in range(TIMESTEP_OUT):
            MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :, :, :], YS_pred[:, i, :, :, :])
            f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (i, name, mode, MSE, RMSE, MAE, MAPE))
            print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (i, name, mode, MSE, RMSE, MAE, MAPE))

def trainModel(name, mode, XS, YS):
    print('Model Training Started ...', time.ctime())
    model = getModel()
    model.compile(loss=LOSS, optimizer=OPTIMIZER)
    model.summary()
    csv_logger = CSVLogger(os.path.join(PATH, f'{name}.log'))
    checkpointer = ModelCheckpoint(filepath=os.path.join(PATH, f'{name}.h5'), verbose=1, save_best_only=True)
    scheduler = LearningRateScheduler(lambda epoch: LEARN)
    model.fit(XS, YS, batch_size=BATCHSIZE, epochs=EPOCH, shuffle=True, 
              callbacks=[csv_logger, checkpointer, scheduler], validation_split=TRAINVALSPLIT)
    YS_pred = model.predict(XS)
    YS, YS_pred = YS * MAX_VALUE, YS_pred * MAX_VALUE
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(os.path.join(PATH, f'{name}_prediction_scores.txt'), 'a') as f:
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (name, mode, MSE, RMSE, MAE, MAPE))                               
    print('Model Training Ended ', time.ctime())

MODELNAME = 'ConvLSTMEncoderDecoder'
KEYWORD = f'Density_{DATANAME}_{MODELNAME}' + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../save/' + KEYWORD
                      
def main():   
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)
    
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    GPU = int(GPU)
    os.environ['PYTHONHASHSEED'] = '0'  # necessary for py3                                       
    np.random.seed(100)
    random.seed(100)
    tf.random.set_seed(100)                          
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[GPU], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[GPU], True)
                                           
    data = np.load(DATAPATH)
    data = data / MAX_VALUE
    train_data = data[:TRAIN_DAYS*DAY_TIMESTAMP, :, :, :]
    test_data = data[TRAIN_DAYS*DAY_TIMESTAMP:, :, :, :]
    print(data.shape, train_data.shape, test_data.shape)
        
    trainXS, trainYS = getXSYS(train_data)
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'TRAIN', trainXS, trainYS)
    
    testXS, testYS = getXSYS(test_data)
    print('TEST XS.shape YS,shape', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'TEST', testXS, testYS)

if __name__ == '__main__':
    main()
