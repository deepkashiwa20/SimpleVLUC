import numpy as np

def evaluate(y_true, y_pred, precision=10):
    return MSE(y_true, y_pred), RMSE(y_true, y_pred), MAE(y_true, y_pred), MAPE(y_true, y_pred)

def MSE(y_true, y_pred):
    mse = np.square(y_pred - y_true)
    mse = np.mean(mse)
    return mse

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# def MAPE(y_true, y_pred, null_val=0):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         if np.isnan(null_val):
#             mask = ~np.isnan(y_true)
#         else:
#             mask = np.not_equal(y_true, null_val)
#         mask = mask.astype('float32')
#         mask /= np.mean(mask)
#         mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
#         mape = np.nan_to_num(mask * mape)
#         return np.mean(mape) * 100    
    
def MAPE(y_true, y_pred, epsilon=1.0):
    return np.mean(np.abs(y_pred - y_true) / np.clip(np.abs(y_true), epsilon, None)) * 100



