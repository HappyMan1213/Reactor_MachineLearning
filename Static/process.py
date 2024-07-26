import pandas as pd

# 读取CSV文件
df = pd.read_csv('F:\Reactor_LSTM\Static\data1.csv')

# 筛选出满足3n条件的行索引（即索引为3的倍数）
# 注意：索引是从0开始的，所以这里使用(df.index + 1) % 3 == 0
rows_to_keep = (df.index + 1) % 5 == 0

# 使用布尔索引保留这些行
df_filtered = df[rows_to_keep]

# 将修改后的DataFrame保存回CSV文件
# 如果你希望保留原始文件，可以保存为一个新的文件名
df_filtered.to_csv('your_file_filtered.csv', index=False)