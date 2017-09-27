# coding: utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
n_i=28*28
n_h1=300
n_h2=100
n_out=10
l_r=0.01
n_epo=50
batch_size=50
X=tf.placeholder(tf.float32,shape=(None,n_i),name="X")
y=tf.placeholder(tf.int64,shape=(None),name="y")
# define neuron layer
# DNN solver with two layers        
with tf.name_scope("DNN"):
    h1=fully_connected(X,n_h1,scope="hidden1")
    h2=fully_connected(h1,n_h2,scope="hidden2")
    logits=fully_connected(h2,n_out,scope="output",activation_fn=None)
# costfunction for DNN: softmax cross_entropy     
with tf.name_scope("loss"):
    xentr=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentr,name="loss")
# DNN GradDescent & training  
with tf.name_scope("train"):
    opt=tf.train.GradientDescentOptimizer(l_r)
    training_op=opt.minimize(loss)
# DNN evaluaters        
with tf.name_scope("eval"):
    correct=tf.nn.in_top_k(logits,y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
#######################################################
##################### START ###########################
#######################################################
#create log file 
logdir="tflogs/r_{}/".format(datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
sum_train=tf.summary.scalar('train',accuracy)
sum_test=tf.summary.scalar('test',accuracy)
fwriter=tf.summary.FileWriter(logdir,tf.get_default_graph())
#Initailaize all varibles
init=tf.global_variables_initializer()
saver=tf.train.Saver()
mnist=input_data.read_data_sets("/home/malu1443/Mheat/Python/Machine/")
with tf.Session() as  ses:
    init.run()
    for ep in range(n_epo):
        for it in range(mnist.train.num_examples // batch_size):
            X_batch,y_batch=mnist.train.next_batch(batch_size)
            ses.run(training_op,feed_dict={X: X_batch,y: y_batch})
        summ=sum_train.eval(feed_dict={X: X_batch,y: y_batch})
        sum2=sum_test.eval(feed_dict={X: mnist.test.images,y: mnist.test.labels})
        fwriter.add_summary(summ,ep)
        fwriter.add_summary(sum2,ep)
    save_path=saver.save(ses,"./my_model_final.ckpt")
fwriter.close()
