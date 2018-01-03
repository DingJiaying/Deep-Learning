# coding=UTF-8
import numpy as np
from blaze.expr.expressions import Label
from anaconda_project.internal.conda_api import result

def shouldStop(oldCentroids, centroids, iterations, maxIt):
     if  iterations > maxIt:
        return True 
     return np.array_equal(oldCentroids, centroids)
 
 
def updateLabels(dataSet, centroids):
      numPoints, numDim = dataSet.shape
      for i in range(0, numPoints):
          dataSet[i, -1] = getLabelFromClosestCentroid(dataSet[i,:-1], centroids)
          
def getLabelFromClosestCentroid(dataSetRow, centroids):
    label = centroids[0, -1]
    minDist = np.linalg.norm(dataSetRow - centroids[0, :-1])
    for i in range(1, centroids.shape[0]):
        dist = np.linalg.norm(dataSetRow - centroids[i, :-1])
        if dist < minDist:
            minDist = dist
            label = centroids[i, -1]
    print "minDist:", minDist
    return Label

def getCentroids(dataSet, k):
    
    result = np.zeros(k, dataSet.shape[1])
    for i in range(1, k+1):
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]
        result[i -1, :-1] = np.mean(oneCluster, axis = 0)
        result[i -1, -1]  = i
        
    return result


def kmeans(X, k, maxIt):  # k是种类
    numPoints, numDim = X.shape  #获取X的行数和列数
    
    dataSet = np.zeros((numPoints, numDim + 1)) #在最后多加1列标签
    dataSet[:, :-1] = X  #最后一列全部都是0
    
    #初始化聚类中心
    centroids = dataSet[np.random.randint(numPoints, size = k), :]
    centroids = dataSet[0:2, :]
    
    centroids[:, -1] = range(1, k+1)#添加不同类的标签
    
    
    iterations = 0 #循环次数
    oldCentroids = None
    
    
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print "iteration: \n", iterations
        print "dataSet: \n", dataSet
        print "centroids: \n", centroids
        
        oldCentroids = np.copy(centroids) #改变一个不会影响到另外一个的值
        iterations +=1
        
        
        updateLabels(dataSet, centroids)
        
        
        centroids = getCentroids(dataSet, k)
    return dataSet 

x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
testX =np.vstack((x1, x2, x3, x4))

result = kmeans(testX, 2, 10)
print "final results:"
print result
            
    
     
    