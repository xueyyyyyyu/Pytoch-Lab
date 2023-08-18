import re
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
from gensim.utils import simple_preprocess


def word2vector(source_file, target_file):
    data = pd.read_csv(source_file)
    processed_texts = []
    # 处理文本
    for text in data['review']:
        # print(text)
        # 去除HTML标签
        cleaned_text = re.sub(r'<.*?>', '', text)
        # 进行文本预处理（去掉标点，小写化等）
        processed_text = simple_preprocess(cleaned_text, min_len=1, deacc=False)
        # print(processed_text)
        processed_texts.append(processed_text)

    model = Word2Vec(sentences=processed_texts, vector_size=100, window=5, min_count=1, workers=4)
    model.save("model/word2vector")

    # 创建一个空列表来存储数据
    data_list = []

    for i, processed_text in enumerate(processed_texts):
        vectors = []
        for word in processed_text:
            vector = model.wv[word]
            vectors.append(vector.tolist())

        label = data.loc[i, 'sentiment']

        # 将向量列表转换为 NumPy 数组
        vector_array = np.array(vectors)

        # 将向量数组和情感标签添加到列表中
        data_list.append([vector_array, label])

    # 创建 DataFrame
    vectors_df = pd.DataFrame(data_list, columns=['vector_array', 'sentiment'])

    # 将 DataFrame 存储为新的 CSV 文件
    vectors_df.to_csv(target_file, index=False)


if __name__ == "__main__":
    word2vector('data/train.csv', 'data/vectors_train.csv')
    word2vector('data/val.csv', 'data/vectors_validate.csv')
    word2vector('data/test.csv', 'data/vectors_test.csv')
