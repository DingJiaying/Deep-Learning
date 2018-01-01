# coding=UTF-8

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer  #sklearn只认识0.1，把原来的十足转化成9位有哪一个哪一个就是1
from NeuralNetwork import NeuralNetwork  
from sklearn.cross_validation import train_test_split#分训练集和测试集，可以用于交运算


digits = load_digits()
X = digits.data
y = digits.target
X -= X.min()#标准化，把数据规范到1-0之间
X /= X.max()

nn = NeuralNetwork([64, 100, 10], 'logistic') #输入层和特征向量的维度是一致的，输出层是和标签的种类一致
X_train, X_test, y_train, y_test = train_test_split(X, y)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
print "start fitting"
nn.fit(X_train, labels_train, epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))
    
print confusion_matrix(y_test, predictions)
print classification_report(y_test, predictions)#评价预测结果

#precision预测为1的时候，确实为1的概率， recal事实是1的时候预测为1的概率