import pandas as pd


def process(source_file, target_file):
    # 读取CSV文件
    data = pd.read_csv(source_file)

    # 替换所有的空格为,
    data['vector_array'] = data['vector_array'].str.replace(r'\s+', ", ", regex=True)
    data['vector_array'] = data['vector_array'].str.replace(r'\[, ', '[', regex=True)

    # 将修改后的数据保存回CSV文件
    data.to_csv(target_file, index=False)


if __name__ == "__main__":
    process('data/vectors_train.csv', 'data/final/train.csv')
    process('data/vectors_validate.csv', 'data/final/validate.csv')
    process('data/vectors_test.csv', 'data/final/test.csv')
