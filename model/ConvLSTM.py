import csv
import numpy as np
import os
import shutil
import sys
import time
from datetime import datetime
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model, Model, Sequential
from keras.layers import Input, merge, TimeDistributed, Flatten, RepeatVector, Reshape, UpSampling2D, concatenate, add, Dropout, Embedding
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler
from Param import *
import Metrics 

def getXSYS_single(data):
    XS, YS = [], []  
    for i in range(data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
        x = data[i:i+TIMESTEP_IN, :, :, :]
        y = data[i+TIMESTEP_IN, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS

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
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), input_shape=(None, HEIGHT, WIDTH, CHANNEL), padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=CHANNEL, kernel_size=(3, 3), padding='same', return_sequences=False, activation='relu'))
    return seq

def testModel(name, mode, XS, YS):
    print('Model Evaluation Started ...', time.ctime())
    assert os.path.exists(os.path.join(PATH, f'{name}.h5')), 'model is not existing'
    model = load_model(os.path.join(PATH, f'{name}.h5'))
    model.summary()

    XS_pred_multi, YS_pred_multi = [XS], []
    for i in range(TIMESTEP_OUT):
        XS_tmp = np.concatenate(XS_pred_multi, axis=1)[:, i:, :, :, :]
        YS_pred = model.predict(XS_tmp)[:, np.newaxis, :, :, :]
        XS_pred_multi.append(YS_pred)
        YS_pred_multi.append(YS_pred)
    YS_pred_multi = np.concatenate(YS_pred_multi, axis=1)
    
    print(XS.shape, YS.shape, YS_pred_multi.shape)
    YS, YS_pred_multi = YS * MAX_VALUE, YS_pred_multi * MAX_VALUE
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred_multi)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred_multi)
    with open(os.path.join(PATH, f'{name}_prediction_scores.txt'), 'a') as f:
        f.write("all steps, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print("all steps, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (name, mode, MSE, RMSE, MAE, MAPE))
    
    with open(os.path.join(PATH, f'{name}_prediction_scores.txt'), 'a') as f:
        for i in range(TIMESTEP_OUT):
            MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :, :, :], YS_pred_multi[:, i, :, :, :])
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
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode='auto')
    model.fit(XS, YS, batch_size=BATCHSIZE, epochs=EPOCH, shuffle=True, 
              callbacks=[csv_logger, checkpointer, scheduler, early_stopping], validation_split=TRAINVALSPLIT)
    YS_pred = model.predict(XS)
    YS, YS_pred = YS * MAX_VALUE, YS_pred * MAX_VALUE
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(os.path.join(PATH, f'{name}_prediction_scores.txt'), 'a') as f:
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (name, mode, MSE, RMSE, MAE, MAPE))                               
    print('Model Training Ended ', time.ctime())

################# Parameter Setting #######################
MODELNAME = 'ConvLSTM'
KEYWORD = f'Density_{DATANAME}_{MODELNAME}' + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../save/' + KEYWORD
################# Parameter Setting #######################
                      
def main():   
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)

    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    os.environ['PYTHONHASHSEED'] = '0'  # necessary for py3                                       
    np.random.seed(100)
    random.seed(100)
    tf.set_random_seed(100)                          
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = GPU
    set_session(tf.Session(graph=tf.get_default_graph(), config=config))
                                           
    data = np.load(DATAPATH)
    data = data / MAX_VALUE
    train_data = data[:TRAIN_DAYS*DAY_TIMESTAMP, :, :, :]
    test_data = data[TRAIN_DAYS*DAY_TIMESTAMP:, :, :, :]
    print(data.shape, train_data.shape, test_data.shape)
        
    trainXS, trainYS = getXSYS_single(train_data)
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'TRAIN', trainXS, trainYS)
    
    testXS, testYS = getXSYS(test_data)
    print('TEST XS.shape YS,shape', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'TEST', testXS, testYS)

if __name__ == '__main__':
    main()
