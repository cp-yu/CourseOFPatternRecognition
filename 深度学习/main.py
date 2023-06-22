from __future__ import print_function
import array
from time import time  # 计入时间
import logging   # 打印进程
import matplotlib.pyplot as plt  # 绘图包
import numpy as np
import csv
# import imageio
# import os
# from skimage import img_as_ubyte

## 机器学习的库
# from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC  #支持向量机

# 将程序进展的信息打印下来
logging.basicConfig(level=logging.INFO,format="%(asctime) %(message)")

# #####################若数据集为图片##########################

# lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)
# # 获取数据集的数量，h值和w值
# n_samples,h,w = lfw_people.images.shape
# # 建立特征向量的矩阵 数据的行
# X = lfw_people.data
# # 向量的维度，每个人的特征值的数
# n_features = X.shape[1]
# y = lfw_people.target
# # 看数据集中有多少人参与数据集中
# target_names = lfw_people.target_names
# # 一共 多少类人脸识别
# n_classes  = target_names.shape[0]
# print("total dataset size:")
# print("n_sample:%d" %(n_samples) )
# print("n_features:%d"%(n_features))
# print("n_classes:%d"%(n_classes))
###################################################################

#####################若数据集为.npy文件###############################
# 用来存放所有录入人脸特征的数组
my_array = np.load('feature_extractByPretrain.npy') # 使用numpy载入npy文件
# 样本数为n_samples,每个样本有特征数为n_features
n_samples,n_features = my_array.shape
# 识别标签(性别）
filename = "faceLabel.csv"
with open(filename,'r') as file:
    reader = csv.DictReader(file)
    column = [row['sex_label'] for row in reader]
# 将标签信息转化为数组存储
y = np.array(column)
target_names = ['male','female']
n_classes = len(target_names)

# 调用数据集，分成训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(my_array,y,test_size=0.25)
# 将数据降低维度，特征值减少，提高预测的准确性
n_components = 150
print("Extracting the top %d eigenfaces from %d faces"%(n_components,x_train.shape[0]))
t0 = time()
# 训练集的特征向量来建模
pca = PCA(n_components=n_components,whiten=True).fit(x_train)
print("done in %0.3fs" %(time()-t0))
# # 提取人脸的照片的特征值
# eigenfaces  = pca.components_.reshape((n_components,h,w))
# 打印信息，进行特征向量的降维
print("projecting the input data on the eigenfaces orthonormal basis")
t0= time()
# 降维测试集和训练集
x_train_pca =pca.transform(x_train)
x_test_pca = pca.transform(x_test)
print("done in %0.3fs" % (time()-t0))

################################建立支持向量机###################################
# 调用支持向量机
print("fitting the classifier to the training set ")
t0=time()
# 参数设置不同的值，尝试不同的值，c是权重，gama表示多少的特征点可以被使用，取用不同的c和伽马组合最好的和函数，
param_grid = {'C': [1e3,5e3,1e4,5e4,1e5] ,"gamma":[0.0001,0.0005,0.001,0.005,0.01,0.1],}
# 建立分类器方程
clf = GridSearchCV(SVC(kernel="rbf",class_weight="balanced"),param_grid)
# 根据训练集的中的值经行建模，找到边际最大的超平面
clf = clf.fit(x_train_pca,y_train)
print("done in %0.3fs"%(time()-t0))
print("best estimator found by grid search：")
print(clf.best_estimator_)

###############################使用支持向量机进行预测#####################################
# 预测
print("predicting people 's sex on the test set")
t0 = time()
y_pred = clf.predict(x_test_pca)
# print(type(y_pred)) # 检查数据类型
# print(type(y_test))
# print(y_pred.dtype)
# print(y_test.dtype)
# 调用 classification_report 来比较真实的y和预测的y
M = classification_report(y_test,y_pred,labels=None,target_names=target_names)
print("the comparation between y_test and y_pred:")
print(M)
# 调用 confusion_matrix 来建立混淆矩阵，对角线的数据越多准确率越高
y_test1 = y_test.astype('float64')  # 统一数据类型
y_pred1 = y_pred.astype('float64')
N = confusion_matrix(y_test1,y_pred1,labels=range(n_classes))
print("the matrix is:")
print(N)
# 画混淆矩阵的图
plt.matshow(N,cmap = plt.cm.Reds)
for i in range(len(N)):
    for j in range(len(N)):
        plt.annotate(N[j,i],xy=(i,j),horizontalalignment = 'center',verticalalignment = 'center')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('The matrix')
plt.show()


# 画图(数据集为图片时，运行该程序效果更好）
# def polt_gallery(images,titles,h,w,n_row=3,n_col=4):
#     # 建立背景图，画布 大小
#     plt.figure(figsize=(1.8*n_col,2.4*n_row))
#     #建立背景布局
#     plt.subplots_adjust(bottom=0,left=0.01,right=0.99,top=0.90,hspace=0.35)
#     for i in range(n_row*n_row):
#         plt.subplot(n_row,n_col,i+1)
#         plt.imshow(images[i].reshape(h,w),cmap=plt.cm.gray)
#         plt.title(titles[i],size=12)
#         plt.xticks(())
#         plt.yticks(())

def title(y_pred,y_test,targetnames,i):
    y_pred2 = list((y_pred).astype('int'))
    y_test2 = list((y_test).astype('int'))
    pred_name = target_names[y_pred2[i]].rsplit(' ', 1)[-1]
    ture_name = targetnames[y_test2[i]].rsplit(' ', 1)[-1]
    return (pred_name,ture_name)

prediction_titles = [title(y_pred,y_test,target_names,i) for i in range(y_pred.shape[0])]
print("the prediction_titles are:")
print(prediction_titles)
print("done in %0.3fs" % (time()-t0))

# 调用函数
# polt_gallery(x_test,prediction_titles,h,w)
# # 打印原图和预测的信息
# eigenfaces_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
# polt_gallery(eigenfaces,eigenfaces_titles,h,w)
# plt.show()

# 打印准确度
print("the accuracy is :")
t0 = time()
score = accuracy_score(y_test, y_pred)
print(score)
print("done in %0.3fs" % (time()-t0))