# coding=UTF-8
import numpy as np
import random

def gradientDescent(x, y, theta, alpha, m, numIterations):   #利用梯度下降算法更新参数值 m总共有多少个实例，numIterations循环多少次，
    xTrans = x.transpose()   #对应公式里边的取转置
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)  #内积的表示
        loss = hypothesis - y
           
        cost = np.sum(loss**2) / (2*m)   #cost函数定义可以变
        print("Iteration %d / cost:%f" % (i, cost))
        
        gradient = np.dot(xTrans, loss)/m
        #更新参数
        theta = theta - alpha * gradient
        
    return theta 

def genData(numPoints, bias, variance):  #函数用于测试数据，多少实例，偏好，方差
    x = np.zeros(shape = (numPoints, 2))
    y = np.zeros(shape = numPoints)
    #一条基础的直线
    for i in range(0, numPoints):
        #偏差数值
        x[i][0] = 1 #第一列都为1
        x[i][1] = i #第二列都为i
        #我们的目标变量
        y[i] = (i + bias) + random.uniform(0, 1)*variance
    return x,y

#100个点，噪声导致偏差为25,方差为10
x, y = genData(100, 25, 10)
# print "x:"
# print x
# print "y:"
# print y 
m, n = np.shape(x) #返回行数和列数
n_y = np.shape(y)
print "x shape:", str(m),str(n)
print "y shape:", n_y

numIterations = 100000
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print theta