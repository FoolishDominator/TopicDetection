# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TopicDetection.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(800, 628)
        self.label1 = QtWidgets.QLabel(Form)
        self.label1.setGeometry(QtCore.QRect(70, 140, 51, 16))
        self.label1.setObjectName("label1")
        self.label2 = QtWidgets.QLabel(Form)
        self.label2.setGeometry(QtCore.QRect(90, 180, 21, 20))
        self.label2.setObjectName("label2")
        self.LineEdit_para1 = QtWidgets.QLineEdit(Form)
        self.LineEdit_para1.setGeometry(QtCore.QRect(120, 140, 113, 21))
        self.LineEdit_para1.setText("")
        self.LineEdit_para1.setObjectName("LineEdit_para1")
        self.LineEdit_para2 = QtWidgets.QLineEdit(Form)
        self.LineEdit_para2.setGeometry(QtCore.QRect(120, 180, 113, 21))
        self.LineEdit_para2.setText("")
        self.LineEdit_para2.setObjectName("LineEdit_para2")
        self.Btn_SelectDoc = QtWidgets.QPushButton(Form)
        self.Btn_SelectDoc.setGeometry(QtCore.QRect(50, 50, 93, 28))
        self.Btn_SelectDoc.setObjectName("Btn_SelectDoc")
        self.Btn_Process = QtWidgets.QPushButton(Form)
        self.Btn_Process.setEnabled(True)
        self.Btn_Process.setGeometry(QtCore.QRect(50, 270, 93, 28))
        self.Btn_Process.setObjectName("Btn_Process")
        self.textBrowser = QtWidgets.QTextBrowser(Form)
        self.textBrowser.setGeometry(QtCore.QRect(260, 60, 521, 551))
        self.textBrowser.setObjectName("textBrowser")
        self.LineEdit_para3 = QtWidgets.QLineEdit(Form)
        self.LineEdit_para3.setGeometry(QtCore.QRect(120, 220, 113, 20))
        self.LineEdit_para3.setObjectName("LineEdit_para3")
        self.label3 = QtWidgets.QLabel(Form)
        self.label3.setGeometry(QtCore.QRect(90, 220, 16, 16))
        self.label3.setObjectName("label3")
        self.Btn_Result = QtWidgets.QPushButton(Form)
        self.Btn_Result.setGeometry(QtCore.QRect(50, 400, 81, 31))
        self.Btn_Result.setObjectName("Btn_Result")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "文本主题检测"))
        self.label1.setText(_translate("Form", "主题数"))
        self.label2.setText(_translate("Form", "α"))
        self.Btn_SelectDoc.setText(_translate("Form", "选择文档"))
        self.Btn_Process.setText(_translate("Form", "开始分析"))
        self.label3.setText(_translate("Form", "β"))
        self.Btn_Result.setText(_translate("Form", "查看结果"))
