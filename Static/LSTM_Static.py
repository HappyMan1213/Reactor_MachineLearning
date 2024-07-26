# python 3.8.18  sklearn 1.0.2  keras 2.4.3  tensorflow 2.3.0  numpy 1.19.1

from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
import numpy as np
import os
import tensorflow as tf

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

def tarin(scaler):
    # 读取数据
    dataset = read_csv('F:\Reactor_LSTM\Static\data3.csv', header=0)
    values = dataset.values

    # 转换为浮点数据
    values = values.astype('float32')

    # 自适应归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # 转换为监督学习格式
    reframed = series_to_supervised(scaled, 0, 2)

    #删除不预测的列 [0,1,2  ，  3,4,5]
    reframed.drop(reframed.columns[[6,7,8,9,10]], axis=1, inplace=True)
    values = reframed.values

    # 划分测试集和训练集
    n_train_hours = 800
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # 切片选择输入列和输出列，选择所有行和除最后一列之外的所有列
    train_X, train_y = train[:, :-1], train[:, -1:]
    test_X, test_y = test[:, :-1], test[:, -1:]

    # 转换为3D张量 [样本数, 每个数组的行和列]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    #———————————————————————————————————————————— LSTM模型 ————————————————————————————————————————————
    # 定义模型超参数
    epochs = 500
    batch_size = 1

    # 模型定义 design network
    model = Sequential()
    model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, activation='tanh'))   #定义隐含层1
    model.add(Dropout(0.01))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))   #定义隐含层3
    model.add(Dropout(0.01))
    model.add(Dense(1))   #定义全连接层输出单元
    model.compile(loss='mse', optimizer='adam')

    # 模型训练 fit network
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 仅使用 CPU
    with tf.device('/CPU:0'):
        history = model.fit(train_X, train_y, batch_size, epochs, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    print(history.history['loss'][-1])
        
    # 进行预测 make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # 预测数据逆缩放 invert scaling for forecast
    inv_yhat = concatenate((test_X[:, :-1], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)     
    inv_yhat = pd.DataFrame(inv_yhat) 
    inv_yhat.to_excel('F:\Reactor_LSTM\Static\Train_predict3.xlsx', index=False)

    #保存模型返回指令
    model.save('F:\Reactor_LSTM\Static\model_Static3.h5')
    print('Train has done')

#———————————————————————————————————————————— 进行静态预测 ————————————————————————————————————————————
# 改dataset了，
def predict_Static(scaler):
    #数据处理
    dataset = read_csv('F:\Reactor_LSTM\Static\Data_predict3.csv', header=0)
    Predict = dataset.values
    Predict = Predict.astype('float32')
    def data(scaler,predict):
        scaled = scaler.fit_transform(predict)
        predict = series_to_supervised(scaled, 0, 2)
        predict.drop(predict.columns[[6, 7, 8, 9, 10]], axis=1, inplace=True)
        predict_x = predict.values
        predict_x = predict_x[:, :-1]
        predict_x = predict_x.reshape((predict_x.shape[0], 1, predict_x.shape[1]))
        return predict_x
    predict_X = data(scaler,Predict)

    #预测与数据后处理 手动循环更新数据
    model = keras.models.load_model('F:\Reactor_LSTM\Static\model_Static3.h5')
    predict_y = model.predict(predict_X)
    predict_X = predict_X.reshape((predict_X.shape[0], predict_X.shape[2]))
    predict_y = concatenate((predict_X[:, :-1], predict_y), axis=1)
    predict_y = scaler.inverse_transform(predict_y)     
    predict_y = pd.DataFrame(predict_y) 
    predict_y.to_excel('F:\Reactor_LSTM\Static\predict_y3.xlsx', index=False)
    print(predict_y)

Tag = input("请输入运行模式（ T/P ）：")  #T表示训练模式，P表示预测模式
scaler = MinMaxScaler(feature_range=(0, 1))

if Tag == 'T' :
    tarin(scaler)

if Tag == 'P' :
    if os.path.exists('F:\Reactor_LSTM\Static\model_Static3.h5'):
        predict_Static(scaler)   
    else :
        print('未进行训练呀')