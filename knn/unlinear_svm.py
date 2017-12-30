# coding=UTF-8
from __future__ import print_function

from time import time #计时
import logging        #打印流程
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from IPython.core.pylabtools import figsize

#print(__dot__)

#
logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(message)s') #把程序进展的信息打印出来


lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4) #下载数据库

n_samples, h, w = lfw_people.images.shape    #访问之后返回三个参数，采样的数量，

X = lfw_people.data   #特征向量的矩阵
n_features = X.shape[1]  #特征向量的维度

y = lfw_people.taget
target_names = lfw_people.taget_names
n_classes = target_names.shape[0]

print("Total dataset sizes:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) #使用函数train_test_split直接将数据分类，返回各类的数据和标签


#降维把特征值的维度降下来
n_components = 150  #组成员素的参数
print("Extracting the top %d eigenfaces from %d faces" 
      % (n_components, X_train.shape[0]))
t0 = time()            #设置初始时间方便计时
pca = RandomizedPCA(n_components = n_components, whiten=True).fit(X, X_train) #pca函数可以把高维的数据转换成低维的
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape(n_components, h, w)  #提取人脸的某些特征值

print("Processing the input data on the  eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_train)
print("done in 50.3fs" % (time() - t0))

print("Fitting the classifier to the training set")
t0 = time()  
prama_grid = {'C':[1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }  #c传入不同的参数
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), prama_grid)  #GridSearchCV将不同的参数组合并选出估计结果最好的参数
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % time( - t0))
print("Best estimator fpund by grid search:")
print(clf.best_estimator_)

#输入测试集进行测试
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in 50.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names = target_names)) #判断预测器额准确性
print(confusion_matrix(y_true, y_pred, labels = range(n_classes))

def plot_gallery(images, titles, h, w, n_row=3, n_col=4): 
    """Helper function to plot a gallery of portraits"""  
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))  
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)  
    for i in range(n_row * n_col):  
        plt.subplot(n_row, n_col, i + 1)  
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)  
        plt.title(titles[i], size=12)  
        plt.xticks(())  
        plt.yticks(())
        
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ',1)[-1] 
    true_name = targer_names[y_test[i]].rsplit(' ',1)[-1]      
    return 'predicted:%s\n ture:   %s' %(pred_name,true_name)

predicted_titles = [title(y_pred, y_test, target_names, i)
                    for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

eigenface_titles = ["eigenface $d" % i for i in range(eighfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show() 

