import numpy as np
import time
import os
import sys
import shutil
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from Param import *
import torch
from torch import nn
import random
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader, Subset
import Metrics
from sklearn.model_selection import train_test_split


class customData(Dataset):
    def __init__(self, data):
        self.data = data
        self.getXSYS()

    def getXSYS(self):
        XS, YS = [], []
        for i in range(self.data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = self.data[i:i + TIMESTEP_IN, :, :, :]
            y = self.data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :, :, :]
            XS.append(x), YS.append(y)
        self.XS, self.YS = np.array(XS), np.array(YS)

        #return XS, YS
    def __len__(self):
        return self.XS.shape[0]
    def __getitem__(self, item):
        return self.XS[item], self.YS[item]

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        # input_dim是每个num_layer的第一个时刻的的输入dim，即channel
        # hidden_dim是每一个num_layer的隐藏层单元，如第一层是64，第二层是128，第三层是128
        # kernel_size是卷积核
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        #self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, #卷积输入的尺寸
                              out_channels=4 * self.hidden_dim,  #因为lstmcell有四个门，隐藏层单元是rnn的四倍
                              kernel_size=self.kernel_size,
                              padding=self.padding)


    def forward(self, input_tensor, cur_state):

        """
        Parameters
        ----------
        input_tensor:
            4-D Tensor either of shape (b, c, h, w)
        cur_state:
            list of two tensors [h_cur, c_cur]
            h_cur: 4-D Tensor either of shape (b, c, h, w)
            c_cur: 4-D Tensor either of shape (b, c, h, w)

        Returns
        -------
        h_next, c_next
        """

        h_cur, c_cur = cur_state

        # 把输入张量与h状态张量沿通道维度串联
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        # conv层的卷积不需要和linear一样，可以是多维的，只要channel数目相同即可
        #combined = combined.clone().detach().requires_grad_(True)
        combined_conv = self.conv(combined)# i门，f门，o门，g门放在一起计算，然后使用split函数把输出4*hidden_dim分割成四个门
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        # torch.nn.Sigmoid()(input)等价于torch.sigmoid(input)
        c_next = f * c_cur + i * g  #下一个细胞状态
        h_next = o * torch.tanh(c_next)  #下一个hc

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim=1, hidden_dim=1, kernel_size=(3,3), num_layers=1,
                 batch_first=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size) #核对尺寸，用的函数是静态方法

        # # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        # # kernel_size==hidden_dim=num_layer的维度，因为要遍历num_layer次
        # kernel_size = self._extend_for_multilayer(kernel_size, num_layers)# 转为列表
        # hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)# 转为列表
        # if not len(kernel_size) == len(hidden_dim) == num_layers:# 判断一致性
        #     raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.kernel_size = kernel_size
        #self.bias = bias
        self.return_all_layers = return_all_layers

        self.conv1 = ConvLSTMCell(input_dim=1,
                                          hidden_dim=32,
                                          kernel_size=self.kernel_size)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = ConvLSTMCell(input_dim=32,
                                          hidden_dim=32,
                                          kernel_size=self.kernel_size)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = ConvLSTMCell(input_dim=32,
                                          hidden_dim=32,
                                          kernel_size=self.kernel_size)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.conv4 = ConvLSTMCell(input_dim=32,
                                          hidden_dim=1,
                                          kernel_size=self.kernel_size)


    def forward(self, input_tensor, pred_len=12):
        """

        Parameters
        ----------
        input_tensor: todo   这里forward有两个输入参数，input_tensor 是一个五维数据
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo  第一次传入hidden_state为none
            None. todo implement stateful  默认刚输入hidden_state为空，等着后面给初始化

        Returns
        -------
        layer_output
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4) #pytorch format N*C*H*W
        b, _, _, h, w = input_tensor.size()

        seq_len = input_tensor.size(1)#取time_step
        cur_layer_input = input_tensor

        # h, c = hidden_state[layer_idx]
        h1, c1 = torch.zeros(b, 32, h, w).to(cuda0, dtype=torch.float32), torch.zeros(b, 32, h, w).to(cuda0, dtype=torch.float32)
        h2, c2 = torch.zeros(b, 32, h, w).to(cuda0, dtype=torch.float32), torch.zeros(b, 32, h, w).to(cuda0, dtype=torch.float32)
        h3, c3 = torch.zeros(b, 32, h, w).to(cuda0, dtype=torch.float32), torch.zeros(b, 32, h, w).to(cuda0, dtype=torch.float32)
        h4, c4 = torch.zeros(b, 1, h, w).to(cuda0, dtype=torch.float32), torch.zeros(b, 1, h, w).to(cuda0, dtype=torch.float32)

        output_inner = []
        for t in range(seq_len):
            h1, c1 = self.conv1(cur_layer_input[:, t, :, :, :], [h1, c1])
            # h1, c1 = self.batchnorm1(h1), self.batchnorm1(c1)

            h2, c2 = self.conv2(h1, [h2, c2])
            # h2, c2 = self.batchnorm2(h2), self.batchnorm2(c2)

        layer_output = torch.unsqueeze(h2, 1)
        layer_output = layer_output.expand(layer_output.size()[0], pred_len, layer_output.size()[2], layer_output.size()[3], layer_output.size()[4])

        for r in range(pred_len):
            h3, c3 = self.conv3(layer_output[:, r, :, :, :], [h3, c3])
            # h3, c3 = self.batchnorm3(h3), self.batchnorm3(c3)
            h4, c4 = self.conv4(h3, [h4, c4])
            output_inner.append(h4)
        # output_inner = []
        # for t in range(seq_len):
        #     h1, c1 = self.conv1(cur_layer_input[:, t, :, :, :], [h1, c1])
        #     # h1, c1 = self.batchnorm1(h1), self.batchnorm1(c1)
        #     h2, c2 = h1, c1
        #     h2, c2 = self.conv2(h1, [h2, c2])
        #     # h2, c2 = self.batchnorm2(h2), self.batchnorm2(c2)
        #
        # layer_output = torch.unsqueeze(h2, 1)
        # layer_output = layer_output.expand(layer_output.size()[0], pred_len, layer_output.size()[2], layer_output.size()[3], layer_output.size()[4])
        #
        # for r in range(pred_len):
        #     h3, c3 = h2, c2
        #     h3, c3 = self.conv3(layer_output[:, r, :, :, :], [h3, c3])
        #     # h3, c3 = self.batchnorm3(h3), self.batchnorm3(c3)
        #     h4, c4 = self.conv4(h3, [h4, c4])
        #     output_inner.append(h4)


        layer_output = torch.stack(output_inner, dim=1)
        return layer_output

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            # cell_list[i]是celllstm的单元，以调用里面的方法
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

print('Model Evaluation Started ...', time.ctime())

MODELNAME = 'ConvLSTMEncoderDecoder'
KEYWORD = f'Density_{DATANAME}_{MODELNAME}' + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../save/' + KEYWORD

np.random.seed(100)
random.seed(100)
torch.manual_seed(seed=100)
cuda0 = torch.device('cuda:0')



def train(trainLoader, testLoader, scaler):
    #model.train()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = ConvLSTM()
    model = model.to(cuda0)
    optim = torch.optim.Adam(model.parameters(), lr= LEARN)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')
    criterion = nn.MSELoss()

    count_patience = 0
    min_loss = 1000

    for i in range(EPOCH):
        train_loss = 0.0
        for data_idx, data_batch in enumerate(trainLoader):
            train_x, train_y = data_batch
            train_x = train_x.to(cuda0, dtype=torch.float32)
            train_y = train_y.to(cuda0, dtype=torch.float32)
            optim.zero_grad()
            output = model(train_x)

            loss = criterion(output, train_y)
            train_loss += loss.item()
            loss.backward() # 每次训练前将梯度重置为0
            optim.step()
        train_loss = train_loss/data_idx


        # MAE = 0.0
        # MSE = 0.0
        # RMSE = 0.0
        # MAPE = 0.0
        # for test_idx, test_data in enumerate(testLoader):
        #     test_x, test_y = test_data
        #     test_x = test_x.to(cuda0, dtype=torch.float32)
        #     test_y = test_y.to(cuda0, dtype=torch.float32)
        #
        #
        #     outputs = model(test_x)
        #     # outputs = outputs.permute(1, 0, 2, 3, 4)
        #     #pred = outputs.argmax(dim=1)
        #     #total_accuracy += torch.eq(outputs, test_y).sum().item()
        #     #acc = total_accuracy / len(test_y)
        #
        #     outputs = outputs.cpu().detach().numpy()
        #     testYS = test_y.cpu().detach().numpy()
        #     testYS_m, outputs_m = scaler.inverse_transform(testYS.reshape(testYS.shape[0]*testYS.shape[1], -1)), scaler.inverse_transform(outputs.reshape(outputs.shape[0]*outputs.shape[1], -1))
        #     testYS_m = testYS_m.reshape(testYS.shape[0], testYS.shape[1], testYS.shape[4], testYS.shape[2], testYS.shape[3])
        #     outputs_m = outputs_m.reshape(outputs.shape[0], outputs.shape[1], outputs.shape[4], outputs.shape[2], outputs.shape[3])
        #
        #     aMSE, aRMSE, aMAE, aMAPE = Metrics.evaluate(testYS_m, outputs_m)
        #     MAE += aMAE #metrics.mean_absolute_error(testYS_m, outputs_m)
        #     MSE += aMSE#metrics.mean_squared_error(testYS_m, outputs_m)
        #     RMSE += aRMSE#np.sqrt(mean_squared_error(testYS_m, outputs_m))
        #     MAPE += aMAPE#metrics.mean_absolute_percentage_error(0.001+ testYS_m.reshape(-1), outputs_m.reshape(-1))
        # MAE = MAE/test_idx
        # MSE = MSE/test_idx
        # RMSE = RMSE/test_idx
        # MAPE = MAPE/test_idx


        val_loss = 0.0
        for test_idx, test_data in enumerate(testLoader):
            test_x, test_y = test_data
            test_x = test_x.to(cuda0, dtype=torch.float32)
            test_y = test_y.to(cuda0, dtype=torch.float32)
            outputs = model(test_x)
            loss = criterion(outputs, test_y)
            val_loss += loss.item()
        val_loss = val_loss/test_idx
        scheduler.step(val_loss)
        # print('learning rate is:', scheduler.optimizer.param_groups[0]['lr'])

        if min_loss > val_loss:
            min_loss = val_loss
            print('Model saving {}'.format(i))
            torch.save(model.state_dict(), PATH + '.pth')#保存训练文件net_params.pkl
        else:
            count_patience += 1

        print(
            f'\r epoch:{i + 1}/{EPOCH} train loss:{train_loss:.3f} val loss: {val_loss:.8f} ', end='')

        if count_patience == PATIENCE:
            count_patience = 0
            break
            # print('Model Training Ended ', time.ctime())

            #state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系

def test(testLoader, scaler):
    #model.train()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = ConvLSTM()
    model.load_state_dict(torch.load(PATH + '.pth'))
    model = model.to(cuda0)

    model.eval()
    with torch.no_grad():
        MAE = 0.0
        MSE = 0.0
        RMSE = 0.0
        MAPE = 0.0
        for test_idx, test_data in enumerate(testLoader):
            test_x, test_y = test_data
            test_x = test_x.to(cuda0, dtype=torch.float32)
            test_y = test_y.to(cuda0, dtype=torch.float32)

            outputs = model(test_x)

            outputs = outputs.cpu().numpy()
            testYS = test_y.cpu().numpy()
            testYS_m, outputs_m = scaler.inverse_transform(testYS.reshape(testYS.shape[0]*testYS.shape[1], -1)), scaler.inverse_transform(outputs.reshape(outputs.shape[0]*outputs.shape[1], -1))
            testYS_m = testYS_m.reshape(testYS.shape[0], testYS.shape[1], testYS.shape[4], testYS.shape[2], testYS.shape[3])
            outputs_m = outputs_m.reshape(outputs.shape[0], outputs.shape[1], outputs.shape[4], outputs.shape[2], outputs.shape[3])

            aMSE, aRMSE, aMAE, aMAPE = Metrics.evaluate(testYS_m, outputs_m)
            MAE += aMAE #metrics.mean_absolute_error(testYS_m, outputs_m)
            MSE += aMSE#metrics.mean_squared_error(testYS_m, outputs_m)
            RMSE += aRMSE#np.sqrt(mean_squared_error(testYS_m, outputs_m))
            MAPE += aMAPE#metrics.mean_absolute_percentage_error(0.001+ testYS_m.reshape(-1), outputs_m.reshape(-1))
        MAE = MAE/test_idx
        MSE = MSE/test_idx
        RMSE = RMSE/test_idx
        MAPE = MAPE/test_idx

        print(
            f'\r test: MAE:{MAE:.8f}  MSE:{MSE:.3f}  ' \
            f'RMSE:{RMSE:.3f} MAPE:{MAPE:.3f}', end='')

        print('Model Testing Ended ', time.ctime())
        np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', testYS)

def train_val_dataset(dataset, val_split=0.10):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)

    scaler = StandardScaler()
    data = np.load(DATAPATH)
    # data = data[0:100, :, :, :]

    data = data.reshape(data.shape[0], -1) #行为data.shape[0]行，列自动算出。data.shape[0]:data第一维的长度。
    data = scaler.fit_transform(data)
    data = data.reshape(-1, HEIGHT, WIDTH, CHANNEL)
    data = np.transpose(data, (0, 3, 1, 2))
    train_data = data[:TRAIN_DAYS * DAY_TIMESTAMP, :, :, :]
    test_data = data[TRAIN_DAYS * DAY_TIMESTAMP:, :, :, :]

    trainvalDataset = customData(train_data)
    trainDataset, valDataset = train_val_dataset(trainvalDataset, 0.20)
    testDataset = customData(test_data)
    trainLoader =  DataLoader(trainDataset, batch_size= BATCHSIZE, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size= BATCHSIZE, shuffle=True)
    testLoader =  DataLoader(testDataset, batch_size= 1, shuffle=False)

    train(trainLoader, valLoader, scaler)
    test(testLoader, scaler)



if __name__ == '__main__':
    main()
