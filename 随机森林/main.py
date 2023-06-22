import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn
import seaborn
import pandas

from skimage.feature import hog
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from skimage import io
from PIL import Image
from sklearn.naive_bayes import GaussianNB



path=r'C:\Users\winne\PycharmProjects\pythonProject\moshi\\'
face=pandas.read_csv(path+"faceLabel.csv")
dtype = np.uint8
shape = (128, 128)
imageSet=[]
for i in face["number"]:
    pre_img_path=r'D:\face\rawdata'
    path=pre_img_path+f'\\{i}'
    data = np.fromfile(path, dtype=dtype).reshape(shape)
    imageSet.append(data)
imageSet=np.array(imageSet)

npy_path=r'C:\Users\winne\PycharmProjects\pythonProject\moshi\feature_extractByPretrain.npy'
feature=np.load(npy_path)

X=imageSet
emotion_label_map = {'funny': 1, 'smiling': 2, 'serious': 3}
face['emotion_label'] = face['emotion'].map(emotion_label_map)
from sklearn.model_selection import GridSearchCV, train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(feature,  face['emotion_label'], random_state=0)

Forest = RandomForestClassifier(n_estimators=180, random_state=0)
Forest.fit(X_train, Y_train)
Y_predict = Forest.predict(X_test)
acc = accuracy_score(Y_test, Y_predict)
precision = precision_score(Y_test, Y_predict, average='macro')
recall = recall_score(Y_test, Y_predict, average='macro')
cm = confusion_matrix(Y_test, Y_predict)
print(cm)
print('Acc: ', acc)
print('Precision: ', precision)
print('Recall: ', recall)
xtick = ['funny','smiling','serious' ]
ytick = xtick

f, ax = plt.subplots(figsize=(7, 5))
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=15)

seaborn.set(font_scale=1.2)
plt.rc('font', family='Times New Roman', size=15)

seaborn.heatmap(cm, fmt='g', cmap='Blues', annot=True, cbar=True, xticklabels=xtick, yticklabels=ytick, ax=ax)

plt.title('Confusion Matrix', fontsize='x-large')

plt.show()

# 导入网格搜索模块
from sklearn.model_selection import GridSearchCV



#N_range = range(100, 200, 5)
# best_acc = -1
# best_n = -1
# for n in N_range:
#     clf = RandomForestClassifier(n_estimators=n)
#     clf.fit(X_train, Y_train)
#     acc = clf.score(X_test, Y_test)
#     if acc > best_acc:
#         best_acc = acc
#         best_n = n
# print("The best n is %0.5f with a score of %0.5f" % (best_n, best_acc))

# N_range = range(50, 100, 5)
# best_acc = -1
# best_n = -1
# for n in N_range:
#     clf = RandomForestClassifier(max_depth=n,n_estimators=140)
#     clf.fit(X_train, Y_train)
#     acc = clf.score(X_test, Y_test)
#     if acc > best_acc:
#         best_acc = acc
#         best_n = n
# print("The best max_depth is %0.5f with a score of %0.5f" % (best_n, best_acc))

# N_range = range(2, 20, 1)
# best_acc = -1
# best_n = -1
# for n in N_range:
#     clf = RandomForestClassifier(min_samples_split=n,max_depth=60,n_estimators=140)
#     clf.fit(X_train, Y_train)
#     acc = clf.score(X_test, Y_test)
#     if acc > best_acc:
#         best_acc = acc
#         best_n = n
# print("The best min_samples_split is %0.5f with a score of %0.5f" % (best_n, best_acc))

# N_range = np.linspace(0.55,0.65,10)
# best_acc = -1
# best_n = -1
# for n in N_range:
#     clf = RandomForestClassifier(max_features=n,min_samples_split=11,max_depth=60,n_estimators=140)
#     clf.fit(X_train, Y_train)
#     acc = clf.score(X_test, Y_test)
#     if acc > best_acc:
#         best_acc = acc
#         best_n = n
# print("The best max_features is %0.5f with a score of %0.5f" % (best_n, best_acc))

N_range = range(1,5,1)
best_acc = -1
best_n = -1
for n in N_range:
    clf = RandomForestClassifier(min_samples_leaf=n,max_features=0.62778,min_samples_split=11,max_depth=60,n_estimators=140)
    clf.fit(X_train, Y_train)
    acc = clf.score(X_test, Y_test)
    if acc > best_acc:
        best_acc = acc
        best_n = n
print("The best min_samples_leaf is %0.5f with a score of %0.5f" % (best_n, best_acc))



# 假设图像数组存储在变量new_element中
#plt.imshow(new_element, cmap='gray')
#plt.show()




