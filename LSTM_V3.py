#相比V2版本，通过控制标识Tag划分了训练和预测模式，可以通过输入预测数据的地址进行预测输出
#将预测结果通过xlsx形式输出，便于绘制图表
# python 3.8.18  sklearn 1.0.2  keras 2.4.3  tensorflow 2.3.0  numpy 1.19.1

from numpy import concatenate
from matplotlib import pyplot as plt
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

Tag = input("请输入运行模式（ T/P ）：")  #T表示训练模式，P表示预测模式


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
n_train = 1400
train = values[:n_train, :]
test = values[n_train:, :]
#print(test.shape)

# 切片选择输入列和输出列，最后一列为输出列
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# 转换为3D张量 [样本数, 每个数组的行和列]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#print(train_X.shape, train_y.shape)


#———————————————————————————————————————————— 贝叶斯优化 ————————————————————————————————————————————


# 定义目标函数 design target 
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

def Opt():
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
    rf_bo.maximize( n_iter = 50 )    # 设置训练批次

    # 绘制优化历史
    target_values = [x['target'] for x in rf_bo.res]
    target_values = pd.DataFrame(target_values)
    target_values.to_excel('F:\CCFL_LSTM\LSTM_Model\OPT.xlsx', index=False)
    # plt.plot(target_values)
    # plt.xlabel('Iteration')
    # plt.ylabel('Target Value')
    # plt.title('Optimization History')
    # plt.show()
    return(rf_bo.max)

# ———————————————————————————————————————————— LSTM模型训练 ————————————————————————————————————————————

def Optimization():
    # 定义模型超参数变量 design over_paramater 
    over_paramater = Opt()
    over_paramater = over_paramater['params']
    return over_paramater

def model_train(over_paramater, train_X, train_y, test_X, test_y):
    # 参数获取 over_paramater get
    Hidden_Layer_Num = over_paramater['Hidden_Layer_Num']
    Hidden_Layer_Nodes = over_paramater['Hidden_Layer_Nodes']
    dropout = over_paramater['dropout']
    epochs = over_paramater['epochs']
    batch_size= over_paramater['batch_size']

    # 模型定义 design network
    model = Sequential()
    Hidden_Layer_Num = int(Hidden_Layer_Num)
    Hidden_Layer_Nodes= int(Hidden_Layer_Nodes)
    epochs = int(epochs)
    batch_size = int(batch_size)
    model.add(LSTM(Hidden_Layer_Nodes, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, activation='tanh'))  #设定初始隐含层
    for i in range(1, Hidden_Layer_Num+1):
        if i < Hidden_Layer_Num:
            model.add(LSTM(Hidden_Layer_Nodes, return_sequences=True, activation='tanh'))   #添加隐含层
        else:
            model.add(LSTM(Hidden_Layer_Nodes, return_sequences=False, activation='tanh'))  #添加隐含层
    model.compile(loss='mae', optimizer='adam')
    model.add(Dropout(dropout))
    model.add(Dense(1))   #定义全连接层输出单元
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

    # 模型训练 fit network
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 仅使用 CPU
    with tf.device('/CPU:0'):
        history = model.fit(train_X, train_y, batch_size, epochs, validation_data=(test_X, test_y), verbose=2, shuffle=True)
        
    # 输出 plot history
    loss = pd.DataFrame(history.history['loss'])
    loss.to_excel('F:\CCFL_LSTM\LSTM_Model\loss.xlsx', index=False)
    val_loss = pd.DataFrame(history.history['val_loss'])
    val_loss.to_excel('F:\CCFL_LSTM\LSTM_Model\val_loss.xlsx', index=False)
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()

    # 进行预测 make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # 预测数据逆缩放 invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    #print(inv_yhat.shape)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    inv_yhat = np.array(inv_yhat)

    # 真实数据逆缩放 invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # 输出真实数据和预测数据
    inv_yhat = pd.DataFrame(inv_yhat)
    inv_yhat.to_excel('F:\CCFL_LSTM\LSTM_Model\Train_predict.xlsx', index=False)
    inv_y = pd.DataFrame(inv_y)
    inv_y.to_excel('F:\CCFL_LSTM\LSTM_Model\Train_true.xlsx', index=False)
    # plt.plot(inv_yhat,label='prediction')
    # plt.plot(inv_y,label='true')
    # plt.legend()
    # plt.show()

    # 返回模型 return model
    return model

# ———————————————————————————————————————————— 预测结果输出 ————————————————————————————————————————————
def predict_data_process(data, scaler):
    dataset = read_csv(data, header=0)
    values = dataset.values
    values = values.astype('float32')
    scaled = scaler.fit_transform(values)

    reframed = series_to_supervised(scaled, 1, 1)
    reframed.drop(reframed.columns[[3,4]], axis=1, inplace=True)   #预测函数必须要有Mflow列

    values = reframed.values
    inv_X = values[:, :-1]
    inv_X = inv_X.reshape((inv_X.shape[0], 1, inv_X.shape[1]))

    return inv_X

def model_predict(model, inv_X):
    yhat = model.predict(inv_X)
    inv_X = inv_X.reshape((inv_X.shape[0], inv_X.shape[2]))

    inv_yhat = concatenate((yhat, inv_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    inv_yhat = np.array(inv_yhat)

    predict = pd.DataFrame(inv_yhat)
    predict.to_excel('F:\CCFL_LSTM\LSTM_Model\predict.xlsx', index=False)

    # plt.plot(inv_yhat,label='prediction')
    # plt.legend()
    # plt.show()

# ———————————————————————————————————————————— 主函数 main program ————————————————————————————————————————————
if Tag == 'T' :
    over_paramater = Optimization()
    model = model_train(over_paramater, train_X, train_y, test_X, test_y)
    model.save('F:\CCFL_LSTM\LSTM_Model\model.h5')

if Tag == 'P' :
    Predict_data = 'F:\CCFL_LSTM\LSTM_Model\Predict.csv'
    inv_X = predict_data_process(Predict_data, scaler)
    if os.path.exists('F:\CCFL_LSTM\LSTM_Model\model.h5'):
        model = keras.models.load_model('F:\CCFL_LSTM\LSTM_Model\model.h5')
        model_predict(model, inv_X)
    else:
        print("未进行训练 没有LSTM模型")