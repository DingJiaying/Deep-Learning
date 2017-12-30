# coding=UTF-8
from sklearn import neighbors#引入数据集
from sklearn import datasets #引入算法

knn = neighbors.KNeighborsClassifier()#实例化KNN分类器

iris = datasets.load_iris() #导入数据集

print(iris)

knn.fit(iris.data, iris.target)  #3FIT都是建立模型两个参数分别为特征值，最后的分类

predictedLabel = knn.predict([[0.1,0.2,0.3,0.4]])#预测最后结果测试用例

print(predictedLabel)