#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:08:06 2020

@author: mtj2301
"""

from sklearn.linear_model import Perceptron
import numpy as np
import time

#create dataset： [3,3],[4,3]是正例；[1,1]是负例
X=np.array([[3,3],[4,3],[1,1]])
y=np.array([1,1,-1])

'''
#Perceptron
time_0=time.time()
ppn=Perceptron(shuffle=False,eta0=0.1) #eta0-learning rate shuffle默认是否重新分类训练集
ppn.fit(X,y)#梯度下降法
dur_1=time.time()-time_0
print(ppn.coef_)
print(ppn.intercept_)
print(dur_1)
'''


'''
#origin Perceptron-Gradient Descent
def orgin_Perceptron(X,y,W,b,eta,iteration):
    update=False
    for i in range(X.shape[0]):#shape输出X的形状，0默认是行；range输出范围，从0开始不包括末位数
        if y[i]*(np.dot(X[i],W)+b)<=0:#dot矩阵乘法
          W+=W+eta*y[i]*X[i].reshape(X[i].shape[0],1) #?reshape(X[i].shape[0],1)
          b+=b+eta*y[i]
          update=True
          iteration+=1
          print(W,b)#'W={},b={}'.format(W,b)
          return W,b,update,iteration
    return W,b,update,iteration
 
#origin Perceptron application
def orgin(X,y,eta):
    W=np.zeros((len(X[0]),1))
    b=0
    iteration=0
    update=False
    W,b,update,iteration==orgin_Perceptron(X,y,W,b,eta,iteration)
    while update==True:
          W,b,update,iteration==orgin_Perceptron(X,y,W,b,eta,iteration)
    return W,b,update,iteration
'''


#Gram Matrix
def Gram_Matrix(X):
    Gram=[]
    for i in range(X.shape[0]):
        Gram_i=[]
        for j in range(X.shape[0]):
            gram=np.dot(X[i],X[j].reshape(X[j].shape[0],1))
            Gram_i.extend(gram)
        Gram.extend(Gram_i)
    return Gram
        

#dual Perceptron
def dual_Perceptron(X,y,alpha,b,eta,iteration,Gram):
    update=False
    for i in range(X.shape[0]):
        temp=0
        for j in range(X.shape[0]):
            temp+= alpha[j][0]*y[j]*Gram[i][j] #对偶和原始的区别 ??为啥运行不出来
        if y[i]*(temp+b)<=0:
           alpha[i]+=eta
           b+=eta*y[i]
           update=True
           iteration+=1
           print('alpha[i]={},b={}'.format(alpha[i],b))
           return alpha,b,update,iteration
    return alpha,b,update,iteration

#dual Perceptron application
def dual(X,y,eta):
    alpha=np.zeros((len(X[0]),1))
    b=0
    iteration=0
    Gram=Gram_Matrix(X)
    alpha,b,update,iteration=dual_Perceptron(X,y,alpha,b,eta,iteration,Gram)
    while update==True:
        apha,b,update,iteration=dual_Perceptron(X,y,alpha,b,eta,iteration,Gram)
    return alpha,b,iteration

    
#Apply 
if __name__=='__main__':
    X=np.array([[3,3],[4,3],[1,1]])
    y=np.array([1,1,-1])
    eta=1
    
'''
#原始形式
time_0=time.time()
W,b,update,iteration=orgin(X,y,eta)
dur_1=time.time()-time_0
print('W={},b={}'.format(W,b))
print('duration={}'.format(dur_1))
'''


#对偶形式
time_1=time.time()
alpha,b,iteration=dual(X,y,eta)
dur_2=time.time()-time_1
print('W={},b={}'.format(alpha,b))
print('duration={}'.format(dur_2))



