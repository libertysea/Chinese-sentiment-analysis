import os
import re
import gensim
import jieba
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Conv1D, Concatenate, Flatten, Dropout, Dense, \
    MaxPooling1D
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.np_utils import to_categorical

from custom_losses import WeightedCrossEntropy, BalancedCrossEntropy


tf.config.list_physical_devices('GPU')


def data_process(path):
    data = pd.read_csv(path, header=0, encoding='utf-8')
    review = data["review"]
    labels = data["label"].tolist()

    # 去除标点
    # 由于数据中存在一条空评价,如果为空,则将评价改为为无
    review_new = list(map(lambda x: "空" if type(x) == float else re.sub(r'[^\w\s\u4e00-\u9fa5]+', ' ', x), review))

    # 中文分词
    cut_review = list(map(lambda x: list(jieba.cut(str(x), cut_all=True)), review_new))

    # 读取停用词表
    f = open('stopwords/stopwords_all.txt', 'r', encoding='UTF-8', errors='ignore')
    stopwords = [line.strip() for line in f.readlines()]
    f.close()
    stop_words = []
    for line in stopwords:
        line = line.rstrip("\n")
        stop_words.append(line)
    stop_words.append(' ')
    stop_words.append('')

    # 对于每条评论，去除其中含有的停用词表中的词
    cleaned_review = list(map(lambda x: list(filter(lambda p: p not in stop_words, x)), cut_review))

    data = pd.DataFrame(columns=['label', 'review'])
    data['label'] = labels
    data['review'] = cleaned_review
    print(data['review'])
    data['data_len'] = data['review'].map(lambda x: len(x))
    print(data['data_len'].describe())
    plt.hist(data['data_len'], bins=30, rwidth=0.9, density=True)
    plt.title("Review Length Histogram")
    plt.show()
    return data['label'], data['review']


def word_vec(label, review):
    # 构建分词器
    tokenizer = Tokenizer()
    # 将所有数据放到分词器里边
    tokenizer.fit_on_texts(review)
    # 构建词汇表(按照词频从大到小排列)
    word_index = tokenizer.word_index
    # 将文本转化为序列
    x_review = tokenizer.texts_to_sequences(review)

    x_review = pad_sequences(x_review, maxlen=seq_len)

    word_vectors = gensim.models.KeyedVectors.load_word2vec_format('corpus/sgns.baidubaike.bigram-char',
                                                                   binary=False,
                                                                   encoding='utf-8')
    # 加载预训练的词向量
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word_vectors:
            embedding_matrix[i] = word_vectors[word]

    x_train, x_test, y_train, y_test = train_test_split(x_review, label, test_size=0.1)  # 划分数据集
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)  # 划分数据集

    # 使用过采样平衡数据集,仅对训练集使用
    # ros = RandomOverSampler()
    # x_train, y_train = ros.fit_resample(x_train, y_train)

    smote = SMOTE(sampling_strategy=0.7)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    # 对标签进行one -Hot编码
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    # y_test = to_categorical(y_test)

    embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                input_length=seq_len, trainable=True)

    return x_train, y_train, x_val, y_val, x_test, y_test, embedding_layer


def text_cnn():
    main_input = Input(shape=(seq_len,), dtype='float64')

    # 输入部分
    embed = embedding_layer(main_input)

    # 卷积层
    cnn1 = Conv1D(
        256,
        3,
        padding='same',
        strides=1,
        activation='relu'
    )(embed)

    cnn1 = MaxPooling1D(pool_size=int(cnn1.shape[1]))(cnn1)
    cnn1 = Flatten()(cnn1)
    cnn1 = Dropout(0.2)(cnn1)

    cnn2 = Conv1D(
        256,
        4,
        padding='same',
        strides=1,
        activation='relu'
    )(embed)
    cnn2 = MaxPooling1D(pool_size=int(cnn2.shape[1]))(cnn2)
    cnn2 = Flatten()(cnn2)
    cnn2 = Dropout(0.2)(cnn2)

    cnn3 = Conv1D(
        256,
        5,
        padding='same',
        strides=1,
        activation='relu'
    )(embed)
    cnn3 = MaxPooling1D(pool_size=int(cnn3.shape[1]))(cnn3)
    cnn3 = Flatten()(cnn3)
    cnn3 = Dropout(0.2)(cnn3)

    cnn = Concatenate(axis=-1)([cnn1, cnn2, cnn3])

    flat = Flatten()(cnn)

    # 在池化层到全连接层之前可以加上dropout防止过拟合
    drop = Dropout(0.2)(flat)

    # 全连接层
    x = Dense(256, activation='relu')(drop)

    # 设置权重,对该层的权重使用L2正则化，目的是防止过拟合。其中0.006是L2正则化的惩罚参数。
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.006))(x)

    # 输出层
    main_output = Dense(2, activation='sigmoid')(x)

    # 实例化模型类
    model = Model(inputs=main_input, outputs=main_output)
    model.compile()
    return model


def model_test(x_test, y_test):
    # 加载模型
    # model = load_model('E:/nltk_w/model/model.h5',custom_objects={'BalancedCrossEntropy': BalancedCrossEntropy},compile=False)
    # model.compile(
    #     optimizer=optimizer,
    #     loss=BalancedCrossEntropy(0.4, 0.6),
    #     metrics=['accuracy'])

    model = load_model('model/model.h5', custom_objects={'WeightedCrossEntropy': WeightedCrossEntropy},
                       compile=False)
    model.compile(
        optimizer=optimizer,
        loss=WeightedCrossEntropy(0.4, 0.6),
        metrics=['accuracy'])
    result = model.predict(x_test)  # 预测样本属于每个类别的概率
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签

    # 将测试集标签转化为列表
    # y_test = y_test.tolist()
    # 预测标签
    y_predict = list(map(float, result_labels))
    # print(y_test)
    # print(y_predict)
    print('准确率', metrics.accuracy_score(y_test, y_predict))

    # 计算评估指标
    precision = precision_score(y_test, y_predict, average='weighted')
    recall = recall_score(y_test, y_predict, average='weighted')
    f1 = f1_score(y_test, y_predict, average='weighted')

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_predict)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", annot_kws={"size": 14})
    plt.xlabel('预测值', fontsize=14)
    plt.ylabel('实际值', fontsize=14)
    plt.title('混淆矩阵', fontsize=16)
    # 为图形添加图例
    import matplotlib.patches as mpatches

    recs = [mpatches.Rectangle((0, 0), 1, 1, fc='cornflowerblue') for i in range(2)]
    plt.legend(recs, [0, 1], loc='upper right', fontsize=12)

    plt.show()


if __name__ == '__main__':
    seq_len = 256
    embedding_dim = 300
    path = "data/ChnSentiCorp_htl_all.csv"
    label, review = data_process(path)
    # print(label)
    # print(review)
    x_train, y_train, x_val, y_val, x_test, y_test, embedding_layer = word_vec(label, review)
    model = text_cnn()

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(
        optimizer=optimizer,
        loss=WeightedCrossEntropy(0.4, 0.6),
        metrics=['accuracy'])

    # 回调函数，用于在训练时保存模型的权重
    save_dir = "model"
    checkpoint_callback = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch}.hdf5'), verbose=1,
                                          save_weights_only=False, save_best_only=False)
    # 模型训练。并指定batch_ size为64，epochs为10
    class_weights = {0: 0.6, 1: 0.4}
    history_cnn = model.fit(x_train, y_train, batch_size=64, class_weight=class_weights, epochs=20,
                            validation_data=(x_val, y_val), shuffle=True,
                            callbacks=[checkpoint_callback])
    model.save('model/model.h5')
    print(history_cnn.history)

    plt.plot(history_cnn.history['loss'], label='Train Loss', color='blue', linestyle='-')
    plt.plot(history_cnn.history['val_loss'], label='Validation Loss', color='red', linestyle='--')
    plt.title('Loss Curves', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.show()

    # 绘制模型训练过程的损失曲线和准确率曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history_cnn.history['loss'], label='Train Loss', color='blue', linestyle='-')
    ax1.plot(history_cnn.history['val_loss'], label='Validation Loss', color='red', linestyle='--')
    ax1.set_title('Loss Curves', fontsize=16)
    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.legend(fontsize=12)

    ax2.plot(history_cnn.history['accuracy'], label='Train Accuracy', color='blue', linestyle='-')
    ax2.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy', color='red', linestyle='--')
    ax2.set_title('Accuracy Curves', fontsize=16)
    ax2.set_xlabel('Epochs', fontsize=14)
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.legend(fontsize=12)
    plt.show()

    model_test(x_test, y_test)
