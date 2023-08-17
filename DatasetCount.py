import pandas as pd
from nltk.tokenize import word_tokenize

# 加载划分后的数据集，假设数据保存在 train.csv、val.csv 和 test.csv 中
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')
test_data = pd.read_csv('test.csv')
sum_data = pd.read_csv('IMDB_Dataset.csv')


# 统计 Positive 和 Negative 数据的规模
def count_labels(data):
    positive_count = data[data['sentiment'] == 'positive'].shape[0]
    negative_count = data[data['sentiment'] == 'negative'].shape[0]
    return positive_count, negative_count


train_positive, train_negative = count_labels(train_data)
val_positive, val_negative = count_labels(val_data)
test_positive, test_negative = count_labels(test_data)
sum_positive, sum_negative = count_labels(sum_data)

print(f"Train Positive: {train_positive}, Negative: {train_negative}")
print(f"Validation Positive: {val_positive}, Negative: {val_negative}")
print(f"Test Positive: {test_positive}, Negative: {test_negative}")
print(f"Sum Positive: {sum_positive}, Negative: {sum_negative}")


# 统计评论的平均长度、最大和最小长度
def compute_lengths(data):
    lengths = data['review'].apply(lambda text: len(word_tokenize(text)))
    average_length = lengths.mean()
    max_length = lengths.max()
    min_length = lengths.min()
    return average_length, max_length, min_length


train_avg_length, train_max_length, train_min_length = compute_lengths(train_data)
val_avg_length, val_max_length, val_min_length = compute_lengths(val_data)
test_avg_length, test_max_length, test_min_length = compute_lengths(test_data)
sum_avg_length, sum_max_length, sum_min_length = compute_lengths(sum_data)

print(f"Train Avg Length: {train_avg_length}, Max Length: {train_max_length}, Min Length: {train_min_length}")
print(f"Validation Avg Length: {val_avg_length}, Max Length: {val_max_length}, Min Length: {val_min_length}")
print(f"Test Avg Length: {test_avg_length}, Max Length: {test_max_length}, Min Length: {test_min_length}")
print(f"Sum Avg Length: {sum_avg_length}, Max Length: {sum_max_length}, Min Length: {sum_min_length}")
