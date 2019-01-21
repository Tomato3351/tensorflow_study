# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 13:10:57 2019

@author: TOMATO
"""

import tensorflow as tf
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
#import cv2


#生成训练数据X
batch_size=16
rdm=RandomState(1)
dataset_size=512
X=rdm.rand(dataset_size,2)
#X=rdm.uniform(0,1,(dataset_size,2))
#生成对应标签Y：x1+x2<1的为正样本
Y=[[int(x1+x2<1)] for (x1,x2) in X]

#绘制训练数据图
#Y_inv=np.invert(Y)+2
Y_np=np.array(Y)
X0=np.delete(X,np.where(Y_np==1)[0],axis=0)
X1=np.delete(X,np.where(Y_np==0)[0],axis=0)
plt.figure('train data',figsize=[9,6],frameon=False)
plt.scatter(X0[:,0],X0[:,1],c='red',alpha=0.5)#透明度
plt.scatter(X1[:,0],X1[:,1],c='green',alpha=0.5)#透明度

#生成测试用数据及正确标签
X_test=RandomState().rand(dataset_size,2)
Y_test=[[int(x1+x2<1)] for (x1,x2) in X_test]


x=tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')


w1=tf.Variable(tf.truncated_normal([2,3],mean=0,stddev=1,seed=1))#标准差为1
w2=tf.Variable(tf.random_normal([3,1],mean=0,stddev=1,seed=1))
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
y=tf.sigmoid(y)#y转换为0到1之间的数值，转换后y代表预测是正样本的概率，1-y代表预测是负样本的概率
cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))
                +(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)  


with tf.Session() as sess:

    init_op=tf.global_variables_initializer()
    sess.run(init_op)

    print('w1=\n',sess.run(w1))
    print('w2=\n',sess.run(w2))
    
    STEPS=10000
    for i in range(STEPS):
        start=(i*batch_size)%dataset_size
#        end=min(start+batch_size,dataset_size)
        end=start+batch_size
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%1000==0:
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("%d training step(s),cross entropy on all data is %g"%(i,total_cross_entropy))

    print ('after training w1=\n',sess.run(w1))
    print ('after training w2=\n',sess.run(w2))

    w1_trained=w1.eval(session=sess)
    w2_trained=sess.run(w2)

    #前向传播预测训练数据X，确定最佳阈值thr
    y_pre_train=sess.run(y,feed_dict={x:X})
    y_pre_test=sess.run(y,feed_dict={x:X_test})
accuracy_max=0
thr_bst=0
for thr in np.arange(min(y_pre_train),max(y_pre_train),0.001):
#for thr in np.arange(-0.9,-0.4,0.001):
    Y_pre=[[int(y>thr)] for y in y_pre_train]
    diff=np.array(Y_pre)-np.array(Y)#预测结果与真实结果差
    accuracy=len(np.where(diff==0)[0])/dataset_size
    if accuracy>accuracy_max:
        accuracy_max=accuracy
        thr_bst=thr
        X_wrong=np.delete(X,np.where(diff==0)[0],axis=0)
print('accuracy_max=\n',accuracy_max,'\nthr_bst=\n',thr_bst)
            
##test结果输出及plot
Y_pre_test=[[int(y>thr_bst)] for y in y_pre_test]
diff_test=np.array(Y_pre_test)-np.array(Y_test)
X_wrong_test=np.delete(X_test,np.where(diff_test==0)[0],axis=0)
accuracy_test=len(np.where(diff_test==0)[0])/dataset_size
print('accuracy_test=\n',accuracy_test)
X0_test=np.delete(X_test,np.where(np.array(Y_pre_test)==1)[0],axis=0)
X1_test=np.delete(X_test,np.where(np.array(Y_pre_test)==0)[0],axis=0)


plt.figure('train data',figsize=[9,6],frameon=False)
plt.scatter(X_wrong[:,0],X_wrong[:,1],c='blue',alpha=0.5)#透明度
#
#plt.figure('test data',figsize=[9,6],frameon=False)
#plt.scatter(X_test[:,0],X_test[:,1],c='blue',alpha=0.5)#透明度  

plt.figure('test result',figsize=[9,6],frameon=False)
plt.scatter(X0_test[:,0],X0_test[:,1],c='red',alpha=0.5)#透明度
plt.scatter(X1_test[:,0],X1_test[:,1],c='green',alpha=0.5)#透明度
plt.scatter(X_wrong_test[:,0],X_wrong_test[:,1],c='blue',alpha=0.5)#透明度

plt.figure('test wrong result',figsize=[9,6],frameon=False)
plt.scatter(X_wrong_test[:,0],X_wrong_test[:,1],c='blue',alpha=0.5)#透明度
plt.xlim(0,1)
plt.ylim(0,1)
plt.show(block=False)












