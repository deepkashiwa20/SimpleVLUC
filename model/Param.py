import pandas as pd

DATANAME = 'Earthquake'
TIMESTEP_IN = 12
TIMESTEP_OUT = 12
HEIGHT = 80
WIDTH = 80
CHANNEL = 1
BATCHSIZE = 4
LEARN = 0.0001
EPOCH = 200
PATIENCE = 10
OPTIMIZER = 'Adam'
LOSS = 'MSE'
TRAINVALSPLIT = 0.2
MAX_VALUE = 868.0 # should be revised to the max_value of your own data.

DATAPATH = '../data/20110301_20110311.npy'
START_DATE, END_DATE = '2011-03-01', '2011-03-11'
TIME_INTERVAL = '5min'
day_timestamps = [i.strftime('%H:%M:%S') for i in pd.date_range("00:00", "23:59", freq=TIME_INTERVAL)]
DAY_TIMESTAMP = len(day_timestamps)
TRAIN_DAYS = 10
