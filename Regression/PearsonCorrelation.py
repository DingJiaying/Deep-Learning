# coding=UTF-8

import numpy as np
from astropy.units import Ybarn
import math

def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffYYBar * diffXXBar)
        varX +=  diffXXBar **2
        varY +=  diffYYBar **2
        
    SST = math.sqrt(varX * varY)
    return SSR/SST

#多重线性回归
def polyfit(x, y, degree):
    results = {}  #把结果存下来
    
    coeffs = np.polyfit(x, y, degree)  #degree表示几重的
    results["polynamial"] = coeffs.tolist()
    
    p = np.poly1d(coeffs)
    
    
    yhat = p(x)
    ybar = np.mean(y)
    ssreg = np.sum((yhat - ybar) **2)
    sstot = np.sum((y - ybar) **2)
    results["determination"] = ssreg / sstot
    
    return results

testX = [1,3,8,7,9]
testY = [10,12,24,21,34]

print "r:", computeCorrelation(testX, testY)        
print "r^2:", str(computeCorrelation(testX, testY)**2)     
# 
# print polyfit(testX, testY, 1)["determination"]