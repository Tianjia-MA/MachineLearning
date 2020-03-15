#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 13:25:09 2020

@author: mtj2301
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd

##import data--如何从excel导入
X_origin=pd.DataFrame(
            [['青年','否','否','一般'],
            ['青年','否','否','好'],
            ['青年','是','否','好'],
            ['青年','是','是','一般'],
            ['青年','否','否','一般'],
            ['中年','否','否','一般'],
            ['中年','否','否','好'],
            ['中年','是','是','好'],
            ['中年','否','是','非常好'],
            ['中年','否','是','非常好'],
            ['老年','否','是','非常好'],
            ['老年','否','是','好'],
            ['老年','是','否','好'],
            ['老年','是','否','非常好'],
            ['老年','否','否','一般']]
            )
#print(X_origin)
Y_origin=pd.DataFrame(['否','否','是','是','否','否','否','是','是','是','是','是','是','是','否'])
#print(Y)
X_origin_test=pd.DataFrame([['青年','否','是','一般'],['中年','是','否','好'],['老年','否','是','一般']])


#文本数据不能直接调用，要预处理
le_X = preprocessing.LabelEncoder().fit(np.unique(X_origin))
X_train=X_origin.apply(le_X.transform)
print(X_train)
le_Y=preprocessing.LabelEncoder().fit(np.unique(Y_origin))
Y_train=Y_origin.apply(le_Y.transform)
print(Y_train)
X_test=X_origin_test.apply(le_X.transform)
print(X_test)


#调包
clf=DecisionTreeClassifier()
clf.fit(X_train,Y_train)
print(clf.predict(X_test))




















    




    






