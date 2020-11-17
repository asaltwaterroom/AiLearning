import numpy as np
import jieba
import pandas as pd
import os
from keras.layers import Input, Embedding, Lambda
from keras.models import Model, load_model
import keras.backend as k


def stopwordslist():  # 设置停用词
    stopwords = []
    if not os.path.exists('./stopword.txt'):
        print('未发现停用词表')
    else:
        stopwords = [line.strip() for line in open('stopword.txt', encoding='UTF-8').readlines()]
    return stopwords


def get_data(filename):
    f = open(filename, 'r', encoding='UTF-8')
    lines = f.readlines()
    sentences = []  # 去停用词后的句子
    data = []
    stopwords = stopwordslist()
    for line in lines:
        data.append(line.strip())  # 原始句子 strip删除字符串头尾的空格和换行符
        sts = list(jieba.cut(line.strip(), cut_all=False))  # 分词后
        splits = []  # 去停用词后
        for w in sts:
            if w not in stopwords:
                splits.append(w)
        sentences.append(splits)
    f.close()
    return data, sentences


def build_dit(sentences):  # 建立词典
    words = {}  # 词频表
    num_sentence = 0  # 句子总数
    total = 0.  # 总词频
    for line in sentences:
        num_sentence += 1
        for w in line:
            if w not in words:
                words[w] = 0
            words[w] += 1
            total += 1
        if num_sentence % 100 == 0:
            print(u'已经找到%s个句子' % num_sentence)

    words = {i: j for i, j in words.items() if j >= min_count}  # 截断词频
    id2word = {i + 1: j for i, j in enumerate(words)}  # id到词语的映射，0表示UNK
    word2id = {j: i for i, j in id2word.items()}  # 词语到id的映射
    num_word = len(words) + 1  # 总词数（算上填充符0）

    subsamples = {i: j / total for i, j in words.items() if j / total > subsample_t}
    subsamples = {i: subsample_t / j + (subsample_t / j) ** 0.5 for i, j in
                  subsamples.items()}  # 这个降采样公式，是按照word2vec的源码来的
    subsamples = {word2id[i]: j for i, j in subsamples.items() if j < 1}  # 降采样表
    return num_sentence, id2word, word2id, num_word, subsamples


def data_generator(word2id, subsamples, data):  # 训练数据生成器
    x, y = [], []
    _ = 0
    for d in data:
        d = [0] * window + [word2id[w] for w in d if w in word2id] + [0] * window
        r = np.random.random(len(d))
        for i in range(window, len(d) - window):
            if d[i] in subsamples and r[i] > subsamples[d[i]]:  # 满足采样条件的直接跳过
                continue
            x.append(d[i - window:i] + d[i + 1:i + 1 + window])
            y.append([d[i]])
        _ += 1
        if _ == num_sentence_per_batch:
            x, y = np.array(x), np.array(y)
            z = np.zeros((len(x), 1))
            return [x, y], z


def build_w2v(word_size, window, num_word, num_negative):
    k.clear_session()  # 清除之前的模型，省得压满内存
    # CBOW输入
    input_words = Input(shape=(window * 2,), dtype='int32')
    input_vecs = Embedding(num_word, word_size, name='word2vec')(input_words)
    print(Embedding(num_word, word_size, name='word2vec'))
    print(input_words)
    print(Embedding(num_word, word_size, name='word2vec')(input_words))
    input_vecs_sum = Lambda(lambda x: k.sum(x, axis=1))(input_vecs)  # CBOW模型，直接将上下文词向量求和

    # 构造随机负样本，与目标组成抽样
    target_word = Input(shape=(1,), dtype='int32')
    negatives = Lambda(lambda x: k.random_uniform((k.shape(x)[0], num_negative), 0, num_word, 'int32'))(target_word)
    samples = Lambda(lambda x: k.concatenate(x))([target_word, negatives])  # 构造抽样，负样本随机抽。负样本也可能抽到正样本，但概率小。

    # 只在抽样内做Dense和sofmax
    softmax_weight = Embedding(num_word, word_size, name='W')(samples)
    softmax_biases = Embedding(num_word, 1, name='b')(samples)
    softmax = Lambda(lambda x:
                     k.softmax((k.batch_dot(x[0], k.expand_dims(x[1], 2)) + x[2])[:, :, 0])
                     )([softmax_weight, input_vecs_sum, softmax_biases])  # 用Embedding层存参数，用K后端实现矩阵乘法，以此复现Dense层的功能

    model = Model(inputs=[input_words, target_word], outputs=softmax)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def most_similar(word2id, w):
    model = load_model('./word2vec.h5', None)
    embeddings = model.get_weights()[0]
    normalize_embeddings = embeddings / (embeddings ** 2).sum(axis=1).reshape((-1, 1)) ** 0.5  # 词向量归一化，即将模定为1
    v = normalize_embeddings[word2id[w]]
    sims = np.dot(normalize_embeddings, v)
    sort = sims.argsort()[::-1]
    sort = sort[sort > 0]
    return [(id2word[i], sims[i]) for i in sort[:k]]


if __name__ == '__main__':
    filename = 'textdata.txt'
    word_size = 60  # 词向量维度
    window = 5
    num_negative = 15
    min_count = 0
    num_worker = 4  # 读取数据的并发数
    num_epoch = 2
    subsample_t = 1e-5  # 词频大于subsample_t的词语，会被降采样，这是提高速度和词向量质量的有效方案
    num_sentence_per_batch = 20  # 目前是以句子为单位作为batch，多少个句子作为一个batch（这样才容易估计训练过程中的steps参数，另外注意，样本数是正比于字数的。）

    data, sentences = get_data(filename)
    num_sentence, id2word, word2id, num_word, subsamples = build_dit(sentences)
    #ipt, opt = data_generator(word2id, subsamples, data)
    #stepsperepoch = int(num_sentence / num_sentence_per_batch),
    model = build_w2v(word_size, window, num_word, num_negative)
    model.fit_generator(data_generator(word2id, subsamples, data),
                        steps_per_epoch=935,
                        epochs=2
                        )
    model.save('word2vec.h5')
    # plot_model(model, to_file='./word2vec.png', show_shapes=True)
    print(pd.Series(most_similar(word2id, '天龙八部')))
