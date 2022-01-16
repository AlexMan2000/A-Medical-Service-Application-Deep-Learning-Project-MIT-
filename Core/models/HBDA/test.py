import json
import pandas as pd
import numpy as np
import sys
import os
import keras
from sklearn.model_selection import train_test_split

dataset1 = pd.read_json("./icliniqQAs.json") #465
dataset2 = pd.read_json("./questionDoctorQAs.json") #5679
dataset3 = pd.read_json("./ehealthforumQAs.json") #171
dataset4 = pd.read_json("./webmdQAs.json") # 23437
questions_list = list(dict(dataset2.question.value_counts()).keys())

import tensorflow as tf
import keras
from gensim.models import KeyedVectors
from keras import initializers as initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Lambda, Bidirectional, TimeDistributed, Dropout, Conv1D, GlobalMaxPool1D,Layer
from keras.layers.merge import multiply, concatenate
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
import itertools

def text_to_word_list(flag, text):  # 文本分词
    text = str(text)
    text = text.lower()

    if flag == 'cn':
        pass
    else:
        # 英文文本下的文本清理规则
        import re
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


def make_w2v_embeddings(flag, word2vec, df, embedding_dim):  # 将词转化为词向量
    vocabs = {}  # 词序号
    vocabs_cnt = 0  # 词个数计数器

    vocabs_not_w2v = {}  # 无法用词向量表示的词
    vocabs_not_w2v_cnt = 0  # 无法用词向量表示的词个数计数器

    # 停用词
    # stops = set(open('data/stopwords.txt').read().strip().split('\n'))

    for index, row in df.iterrows():
        # 打印处理进度
        if index != 0 and index % 1000 == 0:
            print(str(index) + " sentences embedded.")

        for question in ['question1', 'question2']:
            q2n = []  # q2n -> question to numbers representation
            words = text_to_word_list(flag, row[question])

            for word in words:
                # if word in stops:  # 去停用词
                    # continue
                if word not in word2vec and word not in vocabs_not_w2v:  # OOV的词放入不能用词向量表示的字典中，value为1
                    vocabs_not_w2v_cnt += 1
                    vocabs_not_w2v[word] = 1
                if word not in vocabs:  # 非OOV词，提取出对应的id
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    q2n.append(vocabs_cnt)
                else:
                    q2n.append(vocabs[word])
            df.at[index, question + '_n'] = q2n

    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  # 随机初始化一个形状为[全部词个数，词向量维度]的矩阵
    '''
    词1 [a1, a2, a3, ..., a60]
    词2 [b1, b2, b3, ..., b60]
    词3 [c1, c2, c3, ..., c60]
    '''
    embeddings[0] = 0  # 第一行用0填充，因为不存在index为0的词

    for word,index in vocabs.items():
        if word in word2vec:
            embeddings[index] = word2vec[word]
    del word2vec
    return df, embeddings,vocabs


def split_and_zero_padding(df, max_seq_length):  # 调整tokens长度
    print(df["question1_n"])
    print(df["question2_n"])
    # 训练集矩阵转换成字典
    X = {'left': df['question1_n'], 'right': df['question2_n']}

    # 调整到规定长度
    for dataset, side in itertools.product([X], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

    return dataset


class ManDist(Layer):  # 封装成keras层的曼哈顿距离计算

    # 初始化ManDist层，此时不需要任何参数输入
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # 自动建立ManDist层
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # 计算曼哈顿距离
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # 返回结果
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


from keras.layers import *


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        a = K.softmax(K.tanh(K.dot(x, self.W)))
        a = K.permute_dimensions(a, (0, 2, 1))
        outputs = a * inputs
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]



def split_and_zero_padding(df, max_seq_length):  # 调整tokens长度
    # 训练集矩阵转换成字典
    X = {'left': df['question1_n'], 'right': df['question2_n']}

    # 调整到规定长度
    for dataset, side in itertools.product([X], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

    return dataset


def word_to_index(words):
    embedded = [0] * len(words)

    for index, word in enumerate(words):

        try:
            word_index = vocabs[word]
        except:
            word_index = vocabs["what"]
        embedded[index] = word_index

    return np.array(embedded)




# 修改成你的
TRAIN_CSV = r'C:\Users\DELL\Desktop\MIT_Final_Project\Core\data\Train_Test_Data\train.csv'


flag = 'en'

# 这个我另外发送
embedding_path = r'F:\Downloads\GoogleNews-vectors-negative300.bin.gz'
embedding_dim = 300
max_seq_length = 10

# 加载词向量
print("Loading word2vec model(it may takes 2-3 mins) ...")
embedding_dict = KeyedVectors.load_word2vec_format(embedding_path, binary=True)

# 读取并加载训练集
train_df = pd.read_csv(TRAIN_CSV, encoding='gb18030')
for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]

# 将训练集词向量化
train_df, embeddings, vocabs = make_w2v_embeddings(flag, embedding_dict, train_df, embedding_dim=embedding_dim)


# # 分割训练集
X = train_df[['question1_n', 'question2_n']]
Y = train_df['is_duplicate']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2)
X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)


def shared_model_HBDA(_input):
    embedding_layer = Embedding(len(embeddings) + 1,
                                embedding_dim,
                                input_length=max_seq_length)
    embedded_sequences = embedding_layer(_input)
    l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
    l_dense = TimeDistributed(Dense(200))(l_lstm)
    l_att = AttentionLayer()(l_dense)

    return l_att


# 超参
batch_size = 1024
n_epoch = 9
n_hidden = 50

left_input = Input(shape=(max_seq_length,), dtype='float32')
right_input = Input(shape=(max_seq_length,), dtype='float32')
left_sen_representation = shared_model_HBDA(left_input)
right_sen_representation = shared_model_HBDA(right_input)

man_distance = ManDist()([left_sen_representation, right_sen_representation])
sen_representation = concatenate([left_sen_representation, right_sen_representation, man_distance])
similarity = Dense(1, activation='sigmoid')(Dense(2)(Dense(4)(Dense(16)(sen_representation))))
model = Model(inputs=[left_input, right_input], outputs=[similarity])

model.load_weights("./en_SiameseLSTM.h5")
model.summary()



# 测试方法
print("Welcome to the medical QA bot, press q to quit")
quit = False
while not quit:

    X_input = input("User:")
    if X_input == "q":
        print("Goodbye!")
        break
    words = text_to_word_list(flag, X_input)
    input_seq = word_to_index(words)

    count = 0
    prob_list = []
    for index, question in enumerate(questions_list):
        if count < 100:
            words_list = text_to_word_list(flag, question)
            compare = word_to_index(words_list)
            temp_df = pd.DataFrame({"question1_n": [0], "question2_n": [0]}, dtype="object")
            temp_df.at[0, "question1_n"] = input_seq
            temp_df.at[0, "question2_n"] = compare
            temp = split_and_zero_padding(temp_df, 10)
            result = model([temp["left"], temp["right"]])
            prob_list.append((index, result.numpy()[0][0]))
        count += 1
    prob_list.sort(key=lambda x: x[1], reverse=True)

    import random

    top1 = prob_list[:1]
    answer_list = []
    url_list = []
    index = 0
    for candidate in top1:
        question_index = candidate[0]
        question_sentences = list(dataset2[dataset2.question == questions_list[question_index]].answer)
        url_reference = list(dataset2[dataset2.question == questions_list[question_index]].url)
        answer_list.extend(question_sentences)
        url_list.extend(url_reference)

    randindex = random.randint(0, len(answer_list) - 1)
    randindex_url = random.randint(0, len(url_list) - 1)
    answer = answer_list[randindex]
    url = url_list[randindex_url]

    print("Medic:", answer)
    print("Medic:", url)