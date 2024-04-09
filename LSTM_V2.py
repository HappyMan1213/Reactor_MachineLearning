#相比V1版本，添加了贝叶斯超参数优化，对隐含层层数、节点数、epoch、batch_size进行了优化
# python 3.8.18  sklearn 1.0.2  keras 2.4.3  tensorflow 2.3.0  numpy 1.19.1

from numpy import concatenate
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import numpy as np
import os
import tensorflow as tf
from bayes_opt import BayesianOptimization


#———————————————————————————————————————————— 数据处理 ————————————————————————————————————————————

# 转换为监督学习格式 convert series to supervised learning
def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 读取数据
dataset = read_csv('F:\CCFL_LSTM\LSTM_Model\Data.csv', header=0)
values = dataset.values

# 转换为浮点数据
values = values.astype('float32')
#print(values)

# 自适应归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#print(scaled)

# 转换为监督学习格式
reframed = series_to_supervised(scaled, 1, 1)
#print(reframed.head())

#删除不预测的列 [0,1,2  ，  3,4,5] 保留第五列
reframed.drop(reframed.columns[[3,4]], axis=1, inplace=True)
values = reframed.values

# 划分测试集和训练集
n_train_hours = 736
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
#print(test.shape)

# 切片选择输入列和输出列，选择所有行和除最后一列之外的所有列
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# 转换为3D张量 [样本数, 每个数组的行和列]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#print(train_X.shape, train_y.shape)

#———————————————————————————————————————————— Bayesian Optimization ————————————————————————————————————————————


# 定义贝叶斯优化函数 design target 
def rf_cv(Hidden_Layer_Num, Hidden_Layer_Nodes, epochs, batch_size, dropout):
    Opt_model = Sequential()
    Hidden_Layer_Num = int(Hidden_Layer_Num)
    Hidden_Layer_Nodes= int(Hidden_Layer_Nodes)
    epochs = int(epochs)
    batch_size = int(batch_size)
    Opt_model.add(LSTM(Hidden_Layer_Nodes, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, activation='tanh'))  #设定初始隐含层
    for i in range(1, Hidden_Layer_Num+1):
        if i < Hidden_Layer_Num:
            Opt_model.add(LSTM(Hidden_Layer_Nodes, return_sequences=True, activation='tanh'))   #添加隐含层
        else:
            Opt_model.add(LSTM(Hidden_Layer_Nodes, return_sequences=False, activation='tanh'))  #添加隐含层
    Opt_model.compile(loss='mae', optimizer='adam')
    Opt_model.add(Dropout(dropout))
    Opt_model.add(Dense(1))   #定义全连接层输出单元
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
    with tf.device('/CPU:0'):
        result = Opt_model.fit(train_X, train_y, batch_size, epochs, validation_data=(test_X, test_y), shuffle=True)
    mae = -result.history['loss'][-1]
    return mae

# 设定优化参数范围 Set Opt
rf_bo = BayesianOptimization(
    rf_cv,
    {'Hidden_Layer_Num': (1, 5),
    'Hidden_Layer_Nodes': (16, 64),
    'epochs': (5, 10),
    'batch_size': (10, 20),
    'dropout': (0.001, 0.2)}
)

# 优化并输出最优结果 print best result
rf_bo.maximize( n_iter=20 )

# 绘制优化历史
target_values = [x['target'] for x in rf_bo.res]
plt.plot(target_values)
plt.xlabel('Iteration')
plt.ylabel('Target Value')
plt.title('Optimization History')
plt.show()
# print(rf_bo.max)

#———————————————————————————————————————————— LSTM模型 ————————————————————————————————————————————

# # 定义模型超参数变量 design over_paramater 
# Hidden_Layer_Num = 1
# Hidden_Layer_Nodes = 64
# epochs=10
# batch_size=19

# # 模型定义 design network
# model = Sequential()
# model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, activation='tanh'))   #定义隐含层1
# model.add(LSTM(64, return_sequences=True, activation='tanh'))   #定义隐含层2
# model.add(LSTM(64, return_sequences=False, activation='tanh'))   #定义隐含层3
# model.add(Dense(1))   #定义全连接层输出单元
# model.compile(loss='mae', optimizer='adam')

# # 模型训练 fit network
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 仅使用 CPU
# with tf.device('/CPU:0'):
#     history = model.fit(train_X, train_y, batch_size, epochs, validation_data=(test_X, test_y), verbose=2, shuffle=True)
# print(history.history['loss'][-1])
    
# # 输出 plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

# # 进行预测 make a prediction
# yhat = model.predict(test_X)
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# # 预测数据逆缩放 invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# print(inv_yhat.shape)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
# inv_yhat = np.array(inv_yhat)

# # 真实数据逆缩放 invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]

# # 画出真实数据和预测数据
# pyplot.plot(inv_yhat,label='prediction')
# pyplot.plot(inv_y,label='true')
# pyplot.legend()
# pyplot.show()