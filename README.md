# 实验报告

## 数据集划分与统计

1. 将 `IMDB_Dataset.csv` 按照 7:1:2 的比例划分为 `train.csv` 、`val.csv` 和`test.csv`。

2. 分别统计训练集、验证集、测试集中 positive 和 negative 数据的规模，并统计整个数据集中评论的平均长度（词语个数）、最大和最小长度

   ```
   Train Positive: 17411, Negative: 17589
   Validation Positive: 2561, Negative: 2438
   Test Positive: 5028, Negative: 4973
   Sum Positive: 25000, Negative: 25000
   Train Avg Length: 278.9984285714286, Max Length: 2818, Min Length: 8
   Validation Avg Length: 279.49729945989196, Max Length: 2911, Min Length: 22
   Test Avg Length: 281.17408259174084, Max Length: 1513, Min Length: 11
   Sum Avg Length: 279.48348, Max Length: 2911, Min Length: 8
   ```



## 实验结果

1. 在word2vector数据构造的时候丢失了数据，使得词向量数组中出现省略号，不知道应该如何保存数据
2. 后面的数据集读取和训练模型的代码大概完成了，验证模型和测试模型的代码应该还需要修改，但是我模型还没训练出来

