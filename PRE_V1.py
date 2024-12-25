# MLP+LSTM形式 添加slide形式数据
# python 3.8.18  sklearn 1.0.2  keras 2.4.3  tensorflow 2.3.0  numpy 1.19.1

from numpy import concatenate
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import keras
import numpy as np
import os
import tensorflow as tf
from bayes_opt import BayesianOptimization

#———————————————————————————————————————————— 数据处理 ————————————————————————————————————————————

# 读取数据
dataset = read_csv('F:\Reactor_Pre\LSTM\data_break.csv', header=0)
values = dataset.values

# 转换为浮点数据
values = values.astype('float32')

# 自适应归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(values)

# 划分测试集和训练集
n_train_hours = 2250
train = data[:n_train_hours, :]
test = data[n_train_hours:, :]

# 创建数据滑动窗口
def create_sliding_windows(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 0:data.shape[1]])
        Y.append(data[i+window_size, 5])
    return np.array(X), np.array(Y)

window_size = 50
train_X, train_y = create_sliding_windows(train, window_size)
test_X, test_y = create_sliding_windows(test, window_size) 

# 此处不需要转换，因为slide形式数据自身就是3D张量
# 转换为3D张量 [样本数, 每个数组的行和列]
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

#———————————————————————————————————————————— MLP模型 ————————————————————————————————————————————

# 定义模型超参数，便于调试
epochs = 500
batch_size = 1

# 模型定义 design network
model = Sequential()
model.add(Dense(50, input_shape=(window_size, 6), activation='sigmoid'))   #定义隐含层1
model.add(Dropout(0.01))
model.add(LSTM(64, return_sequences=False, activation='relu'))   #定义隐含层3
model.add(Dropout(0.01))
model.add(Dense(1))   #定义全连接层输出单元
model.compile(loss='mse', optimizer='adam')

# 模型训练 fit network
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 仅使用 CPU
with tf.device('/CPU:0'):
    history = model.fit(train_X, train_y, batch_size, epochs, shuffle=True) # 不清楚什么情况，加入validation输出数据格式是错的
print(history.history['loss'][-1])
model.save('F:\Reactor_Pre\MLP\model.h5')

loss = pd.DataFrame(history.history['loss'])
loss.to_excel('F:\Reactor_Pre\MLP\Train_loss.xlsx', index=False)

#———————————————————————————————————————————— 预测和后处理 ————————————————————————————————————————————
# 进行预测 make a prediction
yhat = model.predict(test_X)

# 这里不适用
# 逆slide函数
# def recreate_sliding_windows(data, window_size):
#     re = []
#     for i in range(data.shape[0]):
#         re.append(data[i, 0, 0:data.shape[2]])    
#     return np.array(re)
# re_train_X = recreate_sliding_windows(train_X, window_size)
# print(re_train_X.shape)

# 切片选择输入列和输出列，选择所有行和除最后一列之外的所有列
test_y = test[:, -1]

# 补全预测数据，由于slide_window的缘故
temp = []
for i in range(yhat.shape[0] + window_size):
    if i < window_size:
        temp.append(0)
    if i >= window_size:
        temp.append(yhat[i- window_size, 0:yhat.shape[1]])
yhat = np.array(temp)
yhat = yhat.astype('float32')

# 预测数据逆缩放 invert scaling for forecast
# inv_yhat = test 这块不能这么写，python存在共享内存的问题
inv_yhat = np.zeros_like(test)
for i in range(test.shape[0]):
    inv_yhat[i,  :] = test[i,  :]
    inv_yhat[i, -1] = yhat[i] # 替换预测数据以能够逆缩放
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, -1] # 只选取最后一列 即预测目标

# 真实数据逆缩放 invert scaling for actual
inv_y = test
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -1]

# 画出真实数据和预测数据
inv_yhat = pd.DataFrame(inv_yhat)
inv_y = pd.DataFrame(inv_y)
inv_yhat.to_excel('F:\Reactor_Pre\MLP\Train_predict.xlsx', index=False)
inv_y.to_excel('F:\Reactor_Pre\MLP\Train_true.xlsx', index=False)
plt.plot(inv_yhat,label='prediction')
plt.plot(inv_y,label='true')
plt.legend()
plt.show()