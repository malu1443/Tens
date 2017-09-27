"""This program calulates score on x function"""
        h2=fully_connected(h1,700,scope="hidden_2")
import tensorflow as tf
import numpy as np
import Booking as ma
from datetime import datetime
from tensorflow.contrib.layers import fully_connected
def leaky_relu(z):
    return tf.maximum(0.01*z,z)
def load_data(size):
    C=ma.Data()
    C.load_data(frac=0.2)
    C.set_batch_size(size)
    return C
def main(n_loops):
    """run the NN program """
    batch_size=300
    # read files:
    #definde n,m
    D=load_data(batch_size)
    X=tf.placeholder(tf.float32,shape=(None,D.n),name="X")
    y=tf.placeholder(tf.int64,shape=(None),name="y")
    he=tf.contrib.layers.variance_scaling_initializer()
    with tf.name_scope("DNN"):
        # Network strukture 
        h3=fully_connected(X,3000,scope="hidden_3")
        h4=fully_connected(h3,1500,scope="hidden_4")
        h5=fully_connected(h4,500,scope="hidden_5")
        out=fully_connected(h5,2,scope="output",activation_fn=None)
    with tf.name_scope("loss_function"):
        # Loss function 
        cros_ent=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=out)
        loss=tf.reduce_mean(cros_ent,name="loss")
    with tf.name_scope("Training_function"):
        # grdient training of P : dP/dW
        opt=tf.train.GradientDescentOptimizer(0.008)
        traing_op=opt.minimize(loss)
    with tf.name_scope("Evalutaion"):
        # Evaluate the function
        corr=tf.nn.in_top_k(out,y,1)
        acc=tf.reduce_mean(tf.cast(corr,tf.float32))
    #Program starts
    logdir="Log/{}/".format(datetime.utcnow().strftime("%Y%m%d_%H%M"))
    sc_train=tf.summary.scalar('train',acc)
    sc_test=tf.summary.scalar('test',acc)
    fwriter=tf.summary.FileWriter(logdir,tf.get_default_graph())
    init=tf.global_variables_initializer()
    saver=tf.train.Saver()
    #Run
    with tf.Session() as ses:
        init.run()
        for e in range(n_loops):
            for b in range(D.loops):
                X1,y1=D.next_b()
                ses.run(traing_op,feed_dict={X: X1,y: y1})
            sum_train=sc_train.eval(feed_dict={X: D.X3,y: D.y3})
            sum_test=sc_test.eval(feed_dict={X: D.X2,y: D.y2})
            fwriter.add_summary(sum_train,e)
            fwriter.add_summary(sum_test,e)
        saver_path=saver.save(ses,"./booking_model.ckpt")
if __name__=='__main__':
    main(40)
