# coding=UTF-8
from numpy  import  *

class cluster_node:  #模仿出树结构
    def __init__(self, vec, left = None, right =None, diatance = 0.0, id = None, Count = 1): #vec把每一行看成是向量，left左节点
        self.left = left
        self.right = right
        self.vec = vec
        self.diatance = diatance
        self.Count = Count
        
def L2dist(v1,v2):
    return sqrt(sum((v1 - v2)**2))

def L1dist(v1,v2):
    return sum(abs(v1 - v2))


def hcluster(features, distance=L2dist):
    
    distances = {}#初始化
    currentclustid = -1
    
    #取第i行赋给向量vec
    clust = [cluster_node(array(features[i]), id = i) for i in range(len(features))]  #把每一个实例当作是一个聚类
    
    while len(clust)>1:
        lowestpair = (0,1)
        closest = distances(clust[0].vec, clust[1].vec) #不知道最大还是最小的时候先初始化一个
        
        for i in range(len(clust)):
            for j in range(i +1, len(clust)):
                
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance()
                d = distances[(clust[i].id, clust[j].id)]
                
                if d<closest:
                    closest = d
                    lowestpair = (i,j) #最近的两个下标
                    
        mergeve = [(clust[lowestpair[0].vec[i] + clust[lowestpair[i].vec]]) /2.0  #采用平均值的方法来衡量两者之间的距离
            for i in range(len(clust[0].vec))]
        #创建新节点
        newcluster = cluster_node(array(mergeve), left = clust[lowestpair[0]],
                                  right = clust[lowestpair[1]],
                                  distance = closest,
                                  id  = currentclustid)
        
        currentclustid -=1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)
        
    return clust[0]

def extract_clusters(clust, dist):
    clusters = {}
    if clust.distance<dist :  #dist是阈值
        return [clust]
    else:
        
        cl = []
        cr = []
        if clust.left!=None:
            cl = extract_clusters(clust.left, dist = dist)
        if clust.right!=None:
            cl = extract_clusters(clust.right, dist = dist)
        return cl+cr
    
def get_cluster_elements(clust): #获得树形结构
    
    if clust.id>=0:
        return [clust.id]
    else:
        cl = []
        cr = []
        if clust.left!=None:
            cl = get_cluster_elements(clust.left)
        if clust.right!=None:
            cl = get_cluster_elements(clust.right)
        return cl+cr
    
def printclust(clust, labels = None, n = 0): #打印树形结构
    
    for i in range(n): print ' ',
    if clust.id<0:  #证明当前节点是一个支点
        
        print '-'
    else:           #证明当前节点是一个叶子
        if labels == None:print clust.id
        else: print labels[clust.id]
        
    if clust.left !=None: printclust(clust.left, labels = labels, n = n+1)
    if clust.right !=None: printclust(clust.right, labels = labels, n = n+1)    
    
def getheight(clust):  #找到深度
    
    if clust.left == None  and clust.right == None: return 1 #只有一个原点
    
    return getheight(clust.left) + getheight(clust.right)

def getdepth(clust):
    
    if clust.left ==None and clust.right == None: return 0
    
    return max(getdepth(clust.left),getdepth(clust.right)) + clust.distance
    
    
    
        