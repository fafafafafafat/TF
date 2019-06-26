#encoding:utf-8
#from sklearn.datasets import load_iris
from sklearn import preprocessing
#import imblearn 
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import warnings
'''
这行代码是为了解决报错FutureWarning: 
The default value of gammma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gammae explicitly to 'auto' or 'scale' to avoid this warning.
"avoid this warning.", FutureWarning)
'''
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)#
#rawData = datasets.load_files("./train_0.txt")
rawData1 = pd.read_csv('train_3.txt',encoding = 'utf-8',header = None,names =range(0,12))
rawData2 = pd.read_csv('train_1.txt',encoding = 'utf-8',header = None,names =range(0,12))
#print (rawData.iloc[:,1])
rawData3 = rawData1.sample(n = 300)
#print(pd.concat([rawData3, rawData2], axis=0))
rawData = shuffle(pd.concat([rawData3, rawData2], axis=0))
#print(rawData)
x = rawData.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]]
#print(x)
y = rawData.iloc[:,0]
#print(y)
#---------------------------注意顺序
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 0)
#print(y_train)
#此处采用均值标准差缩放,将标准分公式中的均值改为中位数，将标准差改为绝对偏差(x - np.mean(x) )/np.std
x_scaled_train = preprocessing.scale(x_train)
x_scaled_test = preprocessing.scale(x_test)
#print(x_scaled_test)
'''
标准化后的函数符合正态分布
x_scaled.mean(axis = 0)
x_scaled.std(axis = 0)

estimators = [('reduce_dim', PCA()), ('clf', SVC())]
pipe = Pipeline(estimators)
#print(pipe)
param_grid = dict(reduce_dim__n_components=[2, 5, 10], clf__C=[0.1, 10, 100],clf__gamma=[1,2,3])
grid_search = GridSearchCV(pipe, param_grid=param_grid)
grid_search.fit(x_scaled_train,y_train)
print(grid_search.best_params_)
'''
model = SVC(kernel = 'rbf',decision_function_shape = 'ovo',degree = 2,gamma = 1,coef0 = 0)
model.fit(x_scaled_train,y_train)
#print(model.score(x_scaled_test,y_test))
y_pred = model.predict(x_scaled_test)
tp = 0
fp = 0
fn = 0
tn = 0
for i in y_pred, j in y_test:
    if i == 1:
        if j == 1:
            tp += 1
        else:
            fp += 1
    else:
        if j == 1:
            fn += 1
        else:
            tn += 1
precision_true = tp/(tp + fp)

#model.fit(y_pred0,y_train)
#y_pred = model.predict(y_pred0)
#准确率是分类正确的样本占总样本个数的比例
print(accuracy_score(y_test, y_pred))
#召回率指实际为正的样本中被预测为正的样本所占实际为正的样本的比例。
print(recall_score(y_test, y_pred))
#精确率指模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例
print(precision_score(y_test, y_pred))
#F1 score是精确率和召回率的调和平均值
print(f1_score(y_test, y_pred) )
#保存模型,.pkl文件打开时用open('','rb')
joblib.dump(model,'model.pkl')
#调用时
# model = joblib.load('model.pkl')
