# Reactor_ML
Reactor Parameter Analysis with Machine Learning

1、使用一维热工水力计算程序完成数据生成功能（relap5 mod3.4）
Data_Processing.py 文件用于批量修改relap程序输入卡，并调用relap程序计算，保留计算后的.O文件，删除占用量大的.R文件，需要对记录的参数使用301卡进行小编辑输出，由于3.4运行界面会弹窗口，因此需要根据运行时间定时关闭relap进程
XlsxGet.py 文件用于批量处理.O文件内小编辑输出结果，并转换为对应的excell文件

2、进行reactor parameter预测
LSTM_V1使用了简单的多输入单输出（MutiInput-SingleOut）LSTM网络进行预测
LSTM_V2增加了贝叶斯优化进行网络超参数优化
LSTM_V3进行了集成

3、进行了LSTM单步预测模型的多步预测测试
使用了RU滚动算法，即用上一步的预测输出作为下一步的输入
使用了Static并行算法，即利用不同的预测模型同时对同一输入值进行不同步长的预测

4、在lstm模型的基础上添加了MLP网络结构，提升了曲线峰值的拟合效果
增加了slide形式的数据输入，完成seq2seq模拟
