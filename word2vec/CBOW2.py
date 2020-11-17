import jieba
from collections import Counter
import numpy as np
from keras.layers import Input, Embedding, Lambda
from keras.models import Model
import keras.backend as K
import random
import pandas as pd

word_size = 128  # 词向量维度
window = 5  # 窗口大小
nb_negative = 50  # 随机负采样的样本数
min_count = 10  # 频数少于min_count的词将会被抛弃，低频词类似于噪声，可以抛弃掉
nb_epoch = 2  # 迭代次数


def get_corpus(filename):
    words = []  # 词库，不去重
    corpus = []
    f = open(filename, 'r', encoding='UTF-8')
    lines = f.readlines()
    stopwords = [line.strip() for line in open('stopword.txt', encoding='UTF-8').readlines()]
    for line in lines:
        words += jieba.lcut(line.strip())  # jieba分词，将句子切分为一个个词，并添加到词库中
        for w in list(jieba.cut(line.strip(), cut_all=False)):
            if w not in stopwords:
                corpus.append(w)
    f.close()
    return words, corpus


words, corpus = get_corpus('textdata.txt')
words = dict(Counter(words))
# 这里使用 Counter 来帮助我们对词频进行了统计，并将其转换为字典形式。例如，若原来 words 中数据为['结巴','结巴','喜欢','我','我','我',
# '我']，经过 Counter()的一顿操作之后，就得到了 Counter({'结巴': 2, '喜欢': 1, '我': 4})。再经过 dict()转换，就将其转换为字典形式。
total = sum(words.values())  # 总词频
words = {i: j for i, j in words.items() if j >= min_count}  # 去掉低频词
id2word = {i + 2: j for i, j in enumerate(words)}  # id到词语的映射，习惯将0设置为PAD，1设置为UNK
id2word[0] = 'PAD'
id2word[1] = 'UNK'
word2id = {j: i for i, j in id2word.items()}  # 词语到id的映射
nb_word = len(id2word)  # 总词数


def get_negative_sample(x, word_range, neg_num):
    negs = []
    while True:
        rand = random.randrange(0, word_range)
        if rand not in negs and rand != x:
            negs.append(rand)
        if len(negs) == neg_num:
            return negs


def data_generator():  # 训练数据生成器
    x, y = [], []
    for sentence in corpus:
        sentence = [0] * window + [word2id[w] for w in sentence if w in word2id] + [0] * window
        # 上面这句代码的意思是，因为我们是通过滑窗的方式来获取训练数据的，那么每一句语料的第一个词和最后一个词
        # 如何出现在中心位置呢？答案就是给它padding一下，例如“我/喜欢/足球”，两边分别补窗口大小个pad，得到“pad pad 我 喜欢 足球 pad pad”
        # 那么第一条训练数据的背景词就是['pad', 'pad','喜欢', '足球']，中心词就是'我'
        for i in range(window, len(sentence) - window):
            x.append(sentence[i - window: i] + sentence[i + 1: window + i + 1])
            y.append([sentence[i]] + get_negative_sample(sentence[i], nb_word, nb_negative))

    x, y = np.array(x), np.array(y)
    z = np.zeros((len(x), nb_negative + 1))
    z[:, 0] = 1
    return x, y, z


x, y, z = data_generator()


def build_model():
    # 第一个输入是周围词
    input_words = Input(shape=(window * 2,), dtype='int32')

    # 建立周围词的Embedding层
    input_vecs = Embedding(nb_word, word_size, name='word2vec')(input_words)

    # CBOW模型，直接将上下文词向量求和
    input_vecs_sum = Lambda(lambda x: K.sum(x, axis=1))(input_vecs)

    # 第二个输入，中心词以及负样本词
    samples = Input(shape=(nb_negative + 1,), dtype='int32')

    # 同样的，中心词和负样本词也有一个Emebdding层，其shape为 (?, nb_word, word_size)
    softmax_weights = Embedding(nb_word, word_size, name='W')(samples)
    softmax_biases = Embedding(nb_word, 1, name='b')(samples)

    # 将加和得到的词向量与中心词和负样本的词向量分别进行点乘
    # 注意到使用了K.expand_dims，这是为了将input_vecs_sum的向量推展一维，才能和softmax_weights进行dot
    input_vecs_sum_dot_ = Lambda(lambda x: K.batch_dot(x[0], K.expand_dims(x[1], 2)))([softmax_weights, input_vecs_sum])

    # 然后再将input_vecs_sum_dot_与softmax_biases进行相加，相当于 y = wx+b中的b项
    # 这里又用到了K.reshape，在将向量加和之后，得到的shape是(?, nb_negative+1, 1)，需要将其转换为(?, nb_negative+1)，才能进行softmax计算nb_negative+1个概率值
    add_biases = Lambda(lambda x: K.reshape(x[0] + x[1], shape=(-1, nb_negative + 1)))(
        [input_vecs_sum_dot_, softmax_biases])

    # 这里苏神用了K.softmax，而不是dense(nb_negative+1, activate='softmax')
    # 这是为什么呢？因为dense是先将上一层的张量先进行全联接，再使用softmax，而向下面这样使用K.softmax，就没有了全联接的过程。
    # 实验下来，小编尝试使用dense（activate='softmax')训练出来的效果很差。
    softmax = Lambda(lambda x: K.softmax(x))(add_biases)

    # 编译模型
    model = Model(inputs=[input_words, samples], outputs=softmax)
    # 使用categorical_crossentropy多分类交叉熵作损失函数
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model


model = build_model()
model.fit([x, y], z, epochs=nb_epoch, batch_size=512)
model.save_weights('word2vec.model')
# embedding层的权重永远在模型的第一层
embeddings = model.get_weights()[0]


def most_similar(w):
    print(word2id[w])
    v = embeddings[word2id[w]]
    print(v)
    sims = np.dot(embeddings, v)
    sort = sims.argsort()[::-1]
    sort = sort[sort > 0]
    return [(id2word[i], sims[i]) for i in sort[:10]]


print(pd.Series(most_similar(u'天龙八部')))
