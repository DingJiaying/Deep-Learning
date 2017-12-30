# coding=UTF-8
import numpy as np         #用于数学计算的库
import pylab as pl        
from sklearn import svm

#创建40个线性可区分的点
np.random.seed(0)  #填入值，可以保证是随机产生的统一批值
X = np.r_[np.random.randn(20,2) - [2,2], np.random.randn(20,2) + [2,2]]  #np.random.randn根据正态分布产生一个矩阵， 减去的是均值和方差
Y = [0] *20 + [1] *20

#建立模型
clf = svm.SVC(kernel = 'linear') #调用SVC方程，是一个线性的核函数
clf.fit(X, Y)  #建立模型.fit

#建立超平面的方程
w = clf.coef_[0]  #根据模型利用clf.coef_提取权值
a = -w[0] / w[0]   #把方程 w0*x + w1*y + w2 = 0转换成点斜式
xx = np.linspace(-5,5)  #产生一些连续的x的值
yy = a * xx - (clf.intercept_[0]) / w[1] #.intercept_[0]就是能够取到的偏差，从模型中直接提取的

##求两条边界线
b = clf.support_vectors_[0]  
yy_down = a*xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a *xx + (b[1] - a * b[0])

#画图
pl.plot(xx, yy, 'k--')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_down, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
          s = 80, facecolors = 'none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()

