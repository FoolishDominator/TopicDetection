# -*- coding: utf-8 -*-
import sys

#pyqt相关包
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog
from PyQt5.QtCore import pyqtSignal, QThread
from TopicDetection import Ui_Form
from PyQt5.QtCore import QDir

#gensim包
import gensim
from gensim.models import LdaModel

from pprint import pprint

import os
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#选择文档包
target_path='not_chosen_yet'
# 生成空的pandas表
data = pd.DataFrame(columns=('FileName', 'content'))

n_topics=-1
alpha=-0.01
beta=-0.01

class ProcessThread(QThread):
    signal = pyqtSignal(str)  # 括号里填写信号传递的参数
    signal_topic_word=pyqtSignal(LdaModel,int)
    signal_classify=pyqtSignal(int,pd.DataFrame)
    signal_coherence=pyqtSignal(float)
    current_dir = QDir.currentPath()

    dic_file = "./lda/stop_dic/dict.txt"
    stop_file = "./lda/stop_dic/stopwords.txt"

    def __init__(self):
        super(ProcessThread, self).__init__()


    def run(self):
        """
        进行任务操作，主要的逻辑操作,返回结果
        """
        self.signal.emit("start analyzing...")

        #doc_list保存每篇文章(raw)
        doc_list=[]
        for i in range(len(data.content)):
        #for i in range(5):
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

        #设置模型参数
        chunksize = 2000
        passes = 20
        iterations = 400
        eval_every = None  # Don't evaluate model perplexity, takes too much time.

        # Make a index to word dictionary.
        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token

        if (alpha == -0.01) | (beta == -0.01):
            model = LdaModel(
                corpus=corpus,
                id2word=id2word,
                chunksize=chunksize,
                alpha='auto',
                eta='auto',
                iterations=iterations,
                num_topics=n_topics,
                passes=passes,
                eval_every=eval_every
            )
        else:
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

        # model.save('qzone.model')  # 将模型保存到硬盘

        #主题对应词语
        # pprint(model.print_topics())
        self.signal_topic_word.emit(model,n_topics)

        # #文章对应主题
        # for i in range(len(corpus)):
        #     tem_file_name=data.iloc[i,0]
        #     print(tem_file_name+":")
        #     doc_lda=model[corpus[i]]
        #     print(doc_lda)

        #一致性
        from gensim.models import CoherenceModel
        coherence_model_lda = CoherenceModel(model=model, texts=processed_doc, dictionary=dictionary,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        # print('\nCoherence Score: ', coherence_lda)  # 越高越好
        self.signal_coherence.emit(coherence_lda)

        #打标签(记录在data中)
        topic_id_list=[]
        for i,row in enumerate(model[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            for j, (topic_id, prop_topic) in enumerate(row):
                if j == 0:
                    topic_id_list.append(topic_id)
                else:
                    break

        data.insert(loc=len(data.columns), column='topic_id', value=topic_id_list)

        self.signal_classify.emit(n_topics,data)
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
        if self.LineEdit_para2.text() !='':
            alpha = float(self.LineEdit_para2.text())
        if self.LineEdit_para3.text() != '':
            beta = float(self.LineEdit_para3.text())

        self.thread = ProcessThread()
        self.thread.signal.connect(self.callback)
        self.thread.signal_topic_word.connect(self.print_topic)
        self.thread.signal_classify.connect(self.doc_classify)
        self.thread.signal_coherence.connect(self.print_coherence)
        self.thread.start()


#读取文档集合包中的文件，加载到data中
    def openfile(self):
        global target_path
        global data
        cur_dir=QDir.currentPath()
        target_path=QFileDialog.getExistingDirectory(self,'打开文件夹',cur_dir)
        if target_path:
            files = os.listdir(target_path)

            for i, file in enumerate(files):  # 遍历文件夹
                location = os.path.join(target_path, file)
                # 1
                file_name = file
                # 2
                with open(location, "r", encoding='utf-8') as f:  # 打开文件
                    content = f.read().replace('\n', ' ')
                    f.close()
                data.loc[i] = [file_name, content]

    def showresult(self):
        global target_path
        tmp_path='./'
        if target_path!='not_chosen_yet':
            tmp_path=target_path

        fname,ftype = QFileDialog.getOpenFileName(self, "打开文件", tmp_path,
                                                  "Txt (*.txt)")
        if fname:
            with open(fname, "r", encoding='utf-8') as f:  # 打开文件
                content = f.read()
                f.close()
            self.textBrowser3.setText(str(content))

    def callback(self, msg):
        self.textBrowser.append(str(msg))

    def print_topic(self,model,n_topics):
        topic_list = model.print_topics(n_topics)
        for topic in topic_list:
            self.textBrowser.append("Topic #%d:" % topic[0])
            self.textBrowser.append(str(topic[1]))

    def doc_classify(self,n_topics,data):
        topic_lists = [[] for _ in range(n_topics)]
        for i in range(len(data.content)):
            tmp_id = data.iloc[i, 2]
            tmp_name = data.iloc[i, 0]
            topic_lists[tmp_id].append(tmp_name)

        # print(topic_lists)
        for i, topic in enumerate(topic_lists):
            self.textBrowser2.append("第%d个主题:" % i)
            #print("第%d个主题:" % i)
            tmp_doc=' '
            for doc in topic:
                tmp_doc=tmp_doc+' '+doc
                #print(doc, end=" ")
            self.textBrowser2.append(tmp_doc)

    def print_coherence(self,coherence):
        self.textBrowser.append("Coherence Score:  "+ str(coherence))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    # 将窗口控件显示在屏幕上
    myWin.show()
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
