import pandas as pd
from sklearn.model_selection import train_test_split

# 加载整个数据集
data = pd.read_csv('data/IMDB_Dataset.csv')

# 划分数据集为训练集、验证集和测试集
train_data, temp_data, train_labels, temp_labels = train_test_split(data['review'], data['sentiment'], test_size=0.3, random_state=42)
validation_data, test_data, validation_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.6667, random_state=42)

# 将划分后的数据保存为 CSV 文件
train_df = pd.DataFrame({'review': train_data, 'sentiment': train_labels})
train_df.to_csv('data/split/train.csv', index=False)

validation_df = pd.DataFrame({'review': validation_data, 'sentiment': validation_labels})
validation_df.to_csv('data/split/val.csv', index=False)

test_df = pd.DataFrame({'review': test_data, 'sentiment': test_labels})
test_df.to_csv('data/split/test.csv', index=False)
