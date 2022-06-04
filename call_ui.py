# -*- coding: utf-8 -*-
import sys

#pyqt相关包
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog
from PyQt5.QtCore import pyqtSignal, QThread
from TopicDetection import Ui_Form

#gensim包
import gensim
from gensim.models import LdaModel

from pprint import pprint

import os
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

n_topics=8
alpha=50/n_topics
beta=0.01

class ProcessThread(QThread):
    signal = pyqtSignal(str)  # 括号里填写信号传递的参数
    signal2 =pyqtSignal(LatentDirichletAllocation,object,int)
    signal_topic_word=pyqtSignal(LdaModel,int)

    output_path = 'D:/PycharmProjects/pyqt5/lda/result'
    file_path = 'D:/PycharmProjects/pyqt5/lda/data'

    dic_file = "D:/PycharmProjects/pyqt5/lda/stop_dic/dict.txt"
    stop_file = "D:/PycharmProjects/pyqt5/lda/stop_dic/stopwords.txt"

    def __init__(self):
        super(ProcessThread, self).__init__()


    def run(self):
        """
        进行任务操作，主要的逻辑操作,返回结果
        """
        self.signal.emit("start analysing...")

        os.chdir(self.file_path)
        data = pd.read_excel("data_en.xlsx")  # content type
        os.chdir(self.output_path)

        #doc_list保存每篇文章(raw)
        doc_list=[]
        #for i in range(len(data.content)):
        for i in range(5):
            doc_list.append(data.content[i])

        #processed_doc保存预处理及分词后的文本
        #进行分词、预处理
        processed_doc=[]
        for doc in doc_list:
            #小写
            doc=doc.lower()
            #去除符号
            for c in string.punctuation:
                doc = doc.replace(c, '')
            #分词
            wordLst = nltk.word_tokenize(doc)
            #去除停用词
            with open(ProcessThread.stop_file, 'rb') as fp:
                stopword = fp.read().decode('utf-8')
            stpwrdlst = stopword.splitlines()
            filtered = [w for w in wordLst if w not in stopwords.words('english')]
            filtered2 = [w for w in filtered if w not in stpwrdlst]

            #仅保留名词
            refiltered = nltk.pos_tag(filtered2)
            filtered2 = [w for w, pos in refiltered if pos.startswith('NN')]

            # 词干化
            ps = PorterStemmer()
            filtered = [ps.stem(w) for w in filtered2]
            #处理后文档
            processed_doc.append(filtered)

        #dictionary保存了所有词及对应id
        #corpus保存了语料库（id、词频）向量
        dictionary=gensim.corpora.Dictionary(processed_doc)
        corpus=[dictionary.doc2bow(doc) for doc in processed_doc]


        # # 设置这个能够看到模型的训练进度
        # import logging
        # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



        # Set training parameters.

        chunksize = 2000
        passes = 20
        iterations = 400
        eval_every = None  # Don't evaluate model perplexity, takes too much time.

        # Make a index to word dictionary.
        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token

        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha=alpha,
            eta=beta,
            iterations=iterations,
            num_topics=n_topics,
            passes=passes,
            eval_every=eval_every
        )

        model.save('qzone.model')  # 将模型保存到硬盘

        # top_topics = model.top_topics(corpus)  # , num_words=20)
        #
        # # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        # avg_topic_coherence = sum([t[1] for t in top_topics]) / n_topics
        # print('Average topic coherence: %.4f.' % avg_topic_coherence)
        #

        # pprint(top_topics)

        #主题对应词语
        #topic_list = model.print_topics(n_topics)
        #print(topic_list)
        pprint(model.print_topics())
        self.signal_topic_word.emit(model,n_topics)

        #文章对应主题
        for i in range(len(corpus)):

            print('no.%d doc: '%i)
            doc_lda=model[corpus[i]]
            print(doc_lda)


        #一致性
        from gensim.models import CoherenceModel
        coherence_model_lda = CoherenceModel(model=model, texts=processed_doc, dictionary=dictionary,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)  # 越高越好

#         n_features = 1000  # 提取1000个特征词语
#         tf_vectorizer = CountVectorizer(strip_accents='unicode',
#                                         max_features=n_features,
#                                         stop_words=None,
#                                         max_df=0.5,
#                                         min_df=10)
#         tf = tf_vectorizer.fit_transform(data.content_cutted)
#
# #        n_topics = 8  # 提取8个主题
# #        n_topics = MyMainForm.LineEdit_para1
#         lda = LatentDirichletAllocation(n_components=n_topics, #主题数量
#                                         max_iter=50, #迭代次数
#                                         learning_method='batch',
#                                         doc_topic_prior=alpha,
#                                         topic_word_prior=beta,
#                                         random_state=0)
#         lda.fit(tf)

#         ###########每个主题对应词语
#         n_top_words = 25
#         tf_feature_names = tf_vectorizer.get_feature_names_out()
# #        topic_word = print_top_words(lda, tf_feature_names, n_top_words)
#         self.signal2.emit(lda,tf_feature_names,n_top_words)
#
#         ###########输出每篇文章对应主题
#         topics = lda.transform(tf)
#         topic = []
#         for t in topics:
#             topic.append(list(t).index(np.max(t)))
#         data['topic'] = topic
#         data.to_excel("data_topic.xlsx", index=False)
#
#         ###########困惑度
#         plexs = []
#         n_max_topics = 16
#         for i in range(1, n_max_topics):
#             print(i)
#             lda = LatentDirichletAllocation(n_components=i, max_iter=50,
#                                             learning_method='batch',
#                                             random_state=0)
#             lda.fit(tf)
#             print(lda.perplexity(tf))
#             plexs.append(lda.perplexity(tf))

        self.signal.emit("complete")  # 发射信号


class MyMainForm(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.Btn_SelectDoc.clicked.connect(self.openfile)
        self.Btn_Process.clicked.connect(self.lda_process)
        self.Btn_Result.clicked.connect(self.showresult)
    def lda_process(self):
        global n_topics
        global alpha
        global beta

        n_topics=int(self.LineEdit_para1.text())
        alpha=float(self.LineEdit_para2.text())
        beta=float(self.LineEdit_para3.text())

        self.thread = ProcessThread()
        self.thread.signal.connect(self.callback)
        self.thread.signal2.connect(self.print_topic_word)
        self.thread.signal_topic_word.connect(self.print_topic)
        self.thread.start()

    def callback(self, msg):
        self.textBrowser.append(str(msg))

    def print_topic_word(self,model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            self.textBrowser.append("Topic #%d:" % topic_idx)
            topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            self.textBrowser.append(str(topic_w))

    def print_topic(self,model,n_topics):
        topic_list = model.print_topics(n_topics)
        for topic in topic_list:
            self.textBrowser.append("Topic #%d:" % topic[0])
            self.textBrowser.append(str(topic[1]))


    def openfile(self):
        openfile_name=QFileDialog.getOpenFileName(self, "打开文件", 'c://', '图像文件(*.jpg *.png)')
#        fname, _ = QFileDialog.getOpenFileName(self, "打开文件", '.', '图像文件(*.jpg *.png)')

    def showresult(self):
        os.system('D:/PycharmProjects/pyqt5/lda/result/data_topic.xlsx')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    # 将窗口控件显示在屏幕上
    myWin.show()
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
