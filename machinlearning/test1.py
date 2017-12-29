# coding=UTF-8
from sklearn.feature_extraction import DictVectorizer  #将其他类型的值转化成数字
import csv
from sklearn import preprocessing
from sklearn import tree 
from sklearn.externals.six import StringIO

allElectronicsData = open(r'/home/djy/workspace/machinlearning/AllElectronics.csv','rb')
reader = csv.reader(allElectronicsData) 
headers = reader.next()

print()

featureList = []  #取每一行的特征值
labelList = []    #取最后一列

for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1,len(row) - 1):
        
        rowDict[headers[i]] = row[i]
    
    featureList.append(rowDict)  #把原始类型转化成字典的形式

print(featureList)


vec = DictVectorizer()  #实例化方法
dummyX = vec.fit_transform(featureList).toarray()#把字典形式的数据转化成0,1表示

print("dummyX" + str(dummyX))
#print(vec.get_feature_names(()))

print('labellist:' + str(labelList))


#直接将class型数据转换成0,1
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY" + str(dummyY))


clf = tree.DecisionTreeClassifier(criterion='entropy') #指明用ID3中的信息熵作为评判跟节点的标准，而系统默认的是CARD
clf = clf.fit(dummyX, dummyY)
print("clf" + str(clf))

#代码可视化
with open("allElectronicInformationGainOri.dot",'w') as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file = f) #要还原原来名称

# #对新的数据进行预测
# oneRowX = dummyX[0, :]
# print("oneRowX;" + str(oneRowX))
# 
# newRowX = oneRowX
# 
# newRowX[0] = 1 #青年人改成中年人
# newRowX[1] = 0
# print("newRowX:" + str(newRowX))
# 
# predictedY = clf.predict(newRowX)
# print("predictedY" + str(predictedY))

