# python 3.8.18  sklearn 1.0.2  keras 2.4.3  tensorflow 2.3.0  numpy 1.19.1

from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
import keras
import numpy as np
import os
import tensorflow as tf

#———————————————————————————————————————————— 数据处理 ————————————————————————————————————————————

# 转换为监督学习格式 convert series to supervised learning
def series_to_supervised(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n_in, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n_out)
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
    dataset = read_csv('F:\Reactor_LSTM\Ru\data copy.csv', header=0)
    values = dataset.values

    # 转换为浮点数据
    values = values.astype('float32')

    # 自适应归一化
    scaled = scaler.fit_transform(values)

    # 转换为监督学习格式
    reframed = series_to_supervised(scaled, 0, 2)

    #删除不预测的列 [0,1,2  ，  3,4,5] 保留第五列
    reframed.drop(reframed.columns[[]], axis=1, inplace=True)
    values = reframed.values

    # 划分测试集和训练集
    n_train_hours = 2250
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # 切片选择输入列和输出列，选择所有行和除最后一列之外的所有列
    train_X, train_y = train[:, :-6], train[:, -6:]
    test_X, test_y = test[:, :-6], test[:, -6:]

    # 转换为3D张量 [样本数, 每个数组的行和列]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    #———————————————————————————————————————————— LSTM模型训练 ————————————————————————————————————————————

    # 定义模型超参数
    epochs = 800
    batch_size = 1

    # 模型定义 design network
    model = Sequential()
    model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, activation='sigmoid'))   #定义隐含层1
    model.add(LSTM(64, return_sequences=True, activation='tanh'))   #定义隐含层3
    model.add(LSTM(64, return_sequences=False, activation='tanh'))   #定义隐含层3
    model.add(Dense(6))   #定义全连接层输出单元
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
    inv_yhat = yhat
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -6:]
    inv_yhat = np.array(inv_yhat)

    # 真实数据逆缩放 invert scaling for actual
    test_y = test_y.reshape((len(test_y), 6))
    inv_y = test_y
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -6:]

    # 画出真实数据和预测数据
    inv_yhat = inv_yhat
    inv_y = inv_y
    predict = pd.DataFrame(inv_yhat)
    predict.to_excel('F:\Reactor_LSTM\Ru\predict.xlsx', index=False)
    pyplot.plot(inv_yhat,label='prediction')
    pyplot.plot(inv_y,label='true')
    pyplot.legend()
    pyplot.show()

    #保存模型返回指令
    model.save('F:\Reactor_LSTM\RU\model_RU.h5')
    print('Train has done')


#———————————————————————————————————————————— 进行滚动预测 ————————————————————————————————————————————
def predict_RU(scaler):
    #数据处理
    dataset = read_csv('F:\Reactor_LSTM\Ru\Data_predict.csv', header=0)
    predict = dataset.values
    predict = predict.astype('float32')
    def data(scaler,predict):
        scaled = scaler.fit_transform(predict)
        predict = series_to_supervised(scaled, 0, 2)
        predict.drop(predict.columns[[]], axis=1, inplace=True)
        predict = predict.values
        predict_x = predict[:, :-6]
        predict_X = predict_x.reshape((predict_x.shape[0], 1, predict_x.shape[1]))
        return predict_X
    predict_X = data(scaler,predict)
    predict_value = ['Tempf','P','Core','Mflow','Pre','L_Mf'] 

    #预测与数据后处理 手动循环更新数据
    model = keras.models.load_model('F:\Reactor_LSTM\RU\model_RU.h5')
    predict_y = model.predict(predict_X)
    predict_X = predict_X.reshape((predict_X.shape[0], predict_X.shape[2]))
    predict_y = scaler.inverse_transform(predict_y)
    predict_value = pd.DataFrame(predict_y)
    print(predict_value)


Tag = input("请输入运行模式（ T/P ）：")  #T表示训练模式，P表示预测模式
scaler = MinMaxScaler(feature_range=(0, 1))

if Tag == 'T' :
    tarin(scaler)

if Tag == 'P' :
    if os.path.exists('F:\Reactor_LSTM\RU\model_RU.h5'):
        predict_RU(scaler)   
    else :
        print('未进行训练呀')