import numpy as np

def fitSR(x,y):
    n = len(x)
    dinominator = 0
    numerator = 0
    for i in range(0,n):
        numerator += (x[i] - np.mean(x)) * (y[i] - np.mean(y))
        dinominator += (x[i] - np.mean(x)) **2
    
    print "numerator",numerator
    print "dinominator", dinominator
    b1 = numerator / float(dinominator)
    b0 = np.mean(y) - b1 * np.mean(x)
    
    return b1, b0

def predict(x, b0, b1):
    temp = b0 + x * b1
    return temp

x = [1,3,2,1,3]
y = [14,24,18,17,27]

b1, b0 = fitSR(x,y)
print "b1",b1
print "b0",b0

y_gu = predict(6, b0, b1)

print y_gu