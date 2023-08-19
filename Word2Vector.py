import re
from gensim.models import Word2Vec
import pandas as pd
from gensim.utils import simple_preprocess


def pre_process(source_file, target_file):
    data = pd.read_csv(source_file)

    def pre(text):
        cleaned_text = re.sub(r'<.*?>', '', text)
        processed_text = simple_preprocess(cleaned_text, min_len=1, deacc=False)
        processed_text_string = ' '.join(processed_text)
        return processed_text_string

    data['review'] = data['review'].apply(pre)

    data.to_csv(target_file, index=False)


def word2vector(source_file, target_file):
    data = pd.read_csv(source_file)

    model = Word2Vec(sentences=data['review'], vector_size=100, window=5, min_count=1, workers=4)
    model.save("model/word2vector")

    def get_vector(text):
        vectors = []
        words = text.split()
        for word in words:
            if word in model.wv:
                vectors.append(model.wv[word])
        return vectors

    data['review'] = data['review'].apply(get_vector)

    data.to_csv(target_file, index=False)


def simplify(source_file, target_file):
    # 读取CSV文件
    data = pd.read_csv(source_file)

    # 替换
    data['review'] = data['review'].str.replace("array(", "", regex=False)
    data['review'] = data['review'].str.replace("dtype=float32),", "", regex=False)
    data['review'] = data['review'].str.replace("dtype=float32)", "", regex=False)

    # 将修改后的数据保存回CSV文件
    data.to_csv(target_file, index=False)


if __name__ == "__main__":
    pre_process('data/split/train.csv', 'data/pre/train.csv')
    word2vector('data/pre/train.csv', 'data/vectors/train.csv')
    simplify('data/vectors/train.csv', 'data/final/train.csv')

    pre_process('data/split/val.csv', 'data/pre/validate.csv')
    word2vector('data/pre/validate.csv', 'data/vectors/validate.csv')
    simplify('data/vectors/validate.csv', 'data/final/validate.csv')

    pre_process('data/split/test.csv', 'data/pre/test.csv')
    word2vector('data/pre/test.csv', 'data/vectors/test.csv')
    simplify('data/vectors/test.csv', 'data/final/test.csv')
