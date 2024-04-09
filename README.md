# Reactor_Ml
Reactor Parameter Analysis with Machine Learning

1、使用一维热工水力计算程序完成数据生成功能（relap5 mod3.4）
Data_Processing.py 文件用于批量修改relap程序输入卡，并调用relap程序计算，保留计算后的.O文件，删除占用量大的.R文件，需要对记录的参数使用301卡进行小编辑输出，由于3.4运行界面会弹窗口，因此需要根据运行时间定时关闭relap进程
XlsxGet.py 文件用于批量处理.O文件内小编辑输出结果，并转换为对应的excell文件

2、进行reactor parameter 预测
LSTM_V1使用了简单的多输入单输出（MutiInput-SingleOut）LSTM网络进行预测
LSTM_V2增加了贝叶斯优化进行网络超参数优化
