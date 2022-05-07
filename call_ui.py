# -*- coding: utf-8 -*-
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog
from PyQt5.QtCore import pyqtSignal, QThread
from TopicDetection import Ui_Form

import os
import pandas as pd
import nltk
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

n_topics=8

def textProcessing(text):
    # 小写
    text = text.lower()
    # 去除特殊标点
    for c in string.punctuation:
        text = text.replace(c, '')
    # 分词
    wordLst = nltk.word_tokenize(text)
    # 去除停用词
    with open(Example.stop_file, 'rb') as fp:
        stopword = fp.read().decode('utf-8')
    stpwrdlst = stopword.splitlines()

    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    filtered2 = [w for w in filtered if w not in stpwrdlst]
    # 仅保留名词
    refiltered = nltk.pos_tag(filtered2)
    filtered2 = [w for w, pos in refiltered if pos.startswith('NN')]
    # 词干化
    ps=PorterStemmer()
    filtered=[ps.stem(w) for w in filtered2]

    return " ".join(filtered)


def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)

        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword


class Example(QThread):
    signal = pyqtSignal(str)  # 括号里填写信号传递的参数
    signal2 =pyqtSignal(LatentDirichletAllocation,object,int)

    output_path = 'D:/PycharmProjects/pyqt5/lda/result'
    file_path = 'D:/PycharmProjects/pyqt5/lda/data'

    dic_file = "D:/PycharmProjects/pyqt5/lda/stop_dic/dict.txt"
    stop_file = "D:/PycharmProjects/pyqt5/lda/stop_dic/stopwords.txt"

    def __init__(self):
        super(Example, self).__init__()


    def run(self):
        """
        进行任务操作，主要的逻辑操作,返回结果
        """
        self.signal.emit("start analysing...")

        print(n_topics)
        print(type(n_topics))

        os.chdir(self.file_path)
        data = pd.read_excel("data_en.xlsx")  # content type
        os.chdir(self.output_path)
        data["content_cutted"] = data.content.apply(textProcessing)

        n_features = 1000  # 提取1000个特征词语
        tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                        max_features=n_features,
                                        stop_words=None,
                                        max_df=0.5,
                                        min_df=10)
        tf = tf_vectorizer.fit_transform(data.content_cutted)

#        n_topics = 8  # 提取8个主题
#        n_topics = MyMainForm.LineEdit_para1
        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                        learning_method='batch',
                                        learning_offset=50,
#                                        doc_topic_prior=0.1,
#                                        topic_word_prior=0.01,
                                        random_state=0)
        lda.fit(tf)

        ###########每个主题对应词语
        n_top_words = 25
        tf_feature_names = tf_vectorizer.get_feature_names_out()
#        topic_word = print_top_words(lda, tf_feature_names, n_top_words)
        self.signal2.emit(lda,tf_feature_names,n_top_words)

        ###########输出每篇文章对应主题
        topics = lda.transform(tf)
        topic = []
        for t in topics:
            topic.append(list(t).index(np.max(t)))
        data['topic'] = topic
        data.to_excel("data_topic.xlsx", index=False)

        ###########困惑度
        plexs = []
        n_max_topics = 16
        for i in range(1, n_max_topics):
            print(i)
            lda = LatentDirichletAllocation(n_components=i, max_iter=50,
                                            learning_method='batch',
                                            learning_offset=50,
                                            random_state=0)
            lda.fit(tf)
            print(lda.perplexity(tf))
            plexs.append(lda.perplexity(tf))

        self.signal.emit("complete")  # 发射信号


class MyMainForm(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.Btn_SelectDoc.clicked.connect(self.openfile)
        self.Btn_Process.clicked.connect(self.test)
    def test(self):
        global n_topics

        n_topics=int(self.LineEdit_para1.text())

        self.thread = Example()
        self.thread.signal.connect(self.callback)
        self.thread.signal2.connect(self.print_topic_word)
        self.thread.start()

    def callback(self, msg):
        self.textBrowser.append(str(msg))

    def print_topic_word(self,model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            self.textBrowser.append("Topic #%d:" % topic_idx)
            topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            self.textBrowser.append(str(topic_w))


    def openfile(self):
        openfile_name=QFileDialog.getOpenFileName(self, "打开文件", 'c://', '图像文件(*.jpg *.png)')
        fname, _ = QFileDialog.getOpenFileName(self, "打开文件", '.', '图像文件(*.jpg *.png)')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    # 将窗口控件显示在屏幕上
    myWin.show()
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
