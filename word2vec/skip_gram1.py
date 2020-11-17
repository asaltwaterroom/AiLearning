import jieba
from collections import Counter
import random
import numpy as np
from keras.layers import Input, Embedding, Lambda
from keras.models import Model
import keras.backend as K
import pandas as pd
import re


def clean_chinese_text(text):
    # 保留英文、数字、中文 使用正则表达式
    comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
    return comp.sub('', text)


def get_corpus(filename):
    stopwords = [line.strip() for line in open('bd.txt', encoding='UTF-8').readlines()]
    words, corpus = [], []
    f = open(filename, 'r', encoding='UTF-8')
    lines = f.readlines()
    for line in lines:
        cut_words = list(jieba.cut(line.strip(), cut_all=False))
        for w in cut_words:
            corpus.append(w)
            if w not in stopwords:
                words.append(w) # 结巴分词，将句子分词并添加到词库
    f.close()
    return words, corpus


def set_dic(words, min_count):
    words_dic = dict(Counter(words))  # Counter统计每个词的个数，dict将其转换为字典 词A：词A的个数
    words_dic = {i: j for i, j in words_dic.items() if j > min_count}  # 去掉低频词
    id2word = {i + 2: j for i, j in enumerate(words_dic)}  # id到词语的映射，习惯将0设置为PAD，1设置为UNK
    id2word[0], id2word[1] = 'PAD', 'UNK'
    word2id = {j: i for i, j in id2word.items()}  # 词语到id的映射，将id到词语的映射反过来
    nb_word = len(id2word)  # 总词数，每个词计数一次
    return id2word, word2id, nb_word


def get_negative_sample(x, word_range, neg_num):
    negative_samples = []
    while True:
        rand = random.randrange(0, word_range)
        if rand not in x and rand not in negative_samples:
            negative_samples.append(rand)
        if len(negative_samples) == neg_num:
            return negative_samples


def data_generator(corpus, neg_num, window):
    x, y = [], []
    for sentence in corpus:
        sentence = [0] * window + [word2id[w] for w in sentence if w in word2id] + [0] * window
        for i in range(window, len(sentence) - window):
            x.append(sentence[i])
            surrounding = sentence[i - window : i] + sentence[i + 1 : window + i + 1]
            y.append(surrounding + get_negative_sample(surrounding, nb_word, nb_negative))
    x, y = np.array(x), np.array(y)
    len_samples = len(surrounding) + nb_negative  # 周围词 + 负样本的大小
    z = np.zeros((len(x), len_samples))  # z = np.zeros((len(x), nb_negative+len(surrounding)))
    z[:, :len(surrounding)] = 1
    return x, y, z, len_samples


def build_model():
    K.clear_session()  # 清除之前的模型，省得压满内存
    # 第一个输入是中心词
    input_word = Input(shape=(1,), dtype='int32')

    input_vec = Embedding(nb_word, word_size, name='word2vec')(input_word)

    # 第二个输入，背景词以及负样本词
    samples = Input(shape=(len_samples,), dtype='int32')

    weights = Embedding(nb_word, word_size, name='W')(samples)

    biases = Embedding(nb_word, 1, name='b')(samples)

    input_vec_dot = Lambda(lambda x: K.batch_dot(x[0], K.expand_dims(x[1], 2)))([weights, input_vec])

    # !!!
    add_biases = Lambda(lambda x: K.reshape(x[0] + x[1], shape=(len_samples, -1)))([input_vec_dot, biases])

    sigmoid = Lambda(lambda x: K.sigmoid(x))(add_biases)

    model = Model(inputs=[input_word, samples], outputs=sigmoid)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

    model.summary()

    return model


def most_similar(id2word, embeddings, word):
    v = embeddings[word2id[word]]
    similar = np.dot(embeddings, v)
    sort = [s for s in similar.argsort()[::-1] if s > 0]
    return [(id2word[i], similar[i]) for i in sort[:10]]


# 定义超参
word_size = 128
window = 5
nb_negative = 50  # neg_num
min_count = 10
nb_epoch = 2


words, corpus = get_corpus('text.txt')
id2word, word2id, nb_word = set_dic(words, min_count)

x, y, z, len_samples = data_generator(corpus, nb_negative, window)

model = build_model()
model.fit([x, y], z, epochs=nb_epoch, batch_size=512)
model.save_weights('word2vec.model')
embedding = model.get_weights()[0]
print(pd.Series(most_similar(id2word, embedding, u'天龙八部')))

