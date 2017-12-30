# coding=UTF-8
from sklearn import svm

X = [[2,0],[2,1],[2,2]]
y = [0,0,1]   #分类标记
clf = svm.SVC(kernel = 'linear') #调用SVC方程，是一个线性的核函数
clf.fit(X,y)  #建立模型.fit

print clf

print clf.support_vectors_  #找到支持向量

print clf.support_    #找到支持向量是X中的哪几个

print clf.n_support_   #每一类里面找到了几个支持向量

print clf.predict([[2, .0]])   #预测一个点的分类