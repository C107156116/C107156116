# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 23:56:55 2021

@author: SASAD
"""

import pandas as pd
import numpy as np


from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve


productname='青春露'
raw_df = pd.read_excel(productname+'_data_text_process_test_smote.xls')
raw_df=raw_df.dropna()
needCol = ['label','skin_types','age','好吸收','透亮','保濕','不引起過敏','不油膩','溫和低刺激','不致痘','不黏膩','修護','春','夏','秋','冬']

print(raw_df)
feature_df = raw_df.loc[:,needCol]

#split train test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # new this
#sc = StandardScaler()

np_df = feature_df.values

print(np_df[:,1:], np_df[:,0])
data, label = np_df[:,1:], np_df[:,0]
trainX, testX, trainY, testY = train_test_split(
    data,label, test_size=0.3, random_state=1, stratify=label)
 

#trainX = sc.fit_transform(trainX)
#testX = sc.fit_transform(testX)
'''
#testX = testX[testY == 0] 詳細分析用
#testY = testY[testY == 0]
#逐行test檢查用
a = 0
for i in range(len(testX)):    
    print(lr.predict([testX[i]]), end=' -> ')
    print(testY[i])
    if a==10:
        input()
        a=0
    a+=1    

'''

#create perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline # new this
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
tree = DecisionTreeClassifier(
        criterion='entropy',
        random_state=1,
        max_depth=15,
    )

pipe_lr = make_pipeline( # new this
        StandardScaler(),
        AdaBoostClassifier(
        base_estimator=tree,
        n_estimators=500, #建立500棵樹
        #max_samples=1.0,
        #max_features=1.0,
        learning_rate=1,
        random_state=1
    ),

    )
        
#lda = LDA(n_components=2)
#trainX_ida = lda.fit_transform(trainX, trainY)
#testX_ida = lda.fit_transform(testX, testY)

#lr = LogisticRegression(random_state=1, solver='lbfgs', multi_class='ovr')#C=100.0, 
pipe_lr.fit(trainX, trainY)
import pickle

with open(productname+'.pickle','wb') as f:
     pickle.dump(pipe_lr, f)
#perception predict test label
predY = pipe_lr.predict(testX)
print('test label len: ', len(testY))
print('Misclassified example: %d'%(testY != predY).sum())
#classification accuracy
print()
print('Accuracy: %.3f' % pipe_lr.score(testX, testY))

#print confusion matrix
confmat = confusion_matrix(y_true=testY, y_pred=predY) # new this
print('-------------------------------------------------')
print('___|0|1|(predict)')
print('0: ', confmat[0][0],' | ', confmat[0][1])
print('1: ', confmat[1][0],'| ', confmat[1][1])
print('-------------------------------------------------')

#print other metrics
print('Precsion: {precision_score(testY, predY, average="micro"):.3f}' )
print('Recall:   %.3f'%recall_score(testY, predY, average="micro"))
print('F1:       %.3f'%f1_score(testY, predY, average="micro"))
print('-------------------------------------------------')



import matplotlib.pyplot as plt
def plot_confusion_matrix(confmat):
    fig, ax = plt.subplots(figsize =(2.5, 2.5))
    ax.matshow(confmat, cmap = plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

plot_confusion_matrix(confmat)

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', '^', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

train_sizes, train_scores, test_scores =learning_curve(estimator=pipe_lr,
                               X=trainX,
                               y=trainY,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)



print(test_scores)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')


plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0, 1.05])
#plt.tight_layout()
# plt.savefig('images/06_05.png', dpi=300)
#plt.figure(figsize=(5,5))
plt.show() 
# Training a perceptron model using the standardized training data:
'''
X_combined_std = np.vstack((trainX_ida, testX_ida))
y_combined = np.hstack((trainY, testY))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=pipe_lr, test_idx=range(len(trainX), len(testX)))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/03_01.png', dpi=300)
plt.show()
'''
