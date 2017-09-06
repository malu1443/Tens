# coding: utf-8
import numpy as np
import random_data as rd
import tensorflow as tf
#creats linear data X,XT,y
r=rd.Reg()
r.gen_linedata(n=1000,k=2,m=1)
X=tf.constant(r.xr,dtype=tf.float32,name="X")
y=tf.constant(r.y,dtype=tf.float32,name="y")
XT=tf.transpose(X)
#set random values for Beta and predict y
the=tf.Variable(tf.random_uniform([2,1],-1.0,1.0),name="theta")
y_pred=tf.matmul(X,the,name="predict")
#error and mse
err=y_pred-y
mse=tf.reduce_mean(tf.square(err),name="mse")
###############################################################################
###########################   V0  #############################################
###############################################################################
grad=2/1000*tf.matmul(tf.transpose(X),err)
train_op=tf.assign(the,the-l_r*grad)
init=tf.global_variables_initializer()
with tf.Session() as ses:
    ses.run(init)
    for e in range(1000):
        if e%100==0:
            print(e,mse.eval())
        ses.run(train_op)
    best=the.eval()
print(best)
###############################################################################
###########################   V1  #############################################
###############################################################################
# autoset grad using autodiff.
grad=tf.gradients(mse,[the])[0]
train_op=tf.assign(the,the-l_r*grad)
init=tf.global_variables_initializer()
with tf.Session() as ses:
    ses.run(init)
    for e in range(10000):
        if e%1000==0:
            print(e,mse.eval())
        ses.run(train_op)
    best=the.eval()
print(best)
###############################################################################
###########################   V2  #############################################
###############################################################################
# use gradient Desent optimizer
opt=tf.train.GradientDescentOptimizer(learning_rate=l_r)
train_op=opt.minimize(mse)
with tf.Session() as ses:
    ses.run(init)
    for e in range(10000):
        if e%1000==0:
            print(e,mse.eval(),grad.eval())
        ses.run(train_op)
    best=the.eval()
    
