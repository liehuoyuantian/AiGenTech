# --*-- coding:utf-8 --*--
'''
Created on 2018年4月18日

@author: Administrator
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from math import sqrt
import numpy as np
import os

# 指定使用GPU
#os.environ['CUDA_VISIBLE_DEVICES']='2,3'

data=input_data.read_data_sets('data/fashion')
'''
data.train.images data.train.labels
data.test.images/labels
data.train.next_batch()
'''
batch_size=32
epochs=20
save_path='./logs/fashion-model.ckpt'
global_step=tf.Variable(1)
learning_rate=tf.train.exponential_decay(learning_rate=0.01, global_step=global_step, decay_steps=1000, decay_rate=0.8, staircase=True)
images=tf.placeholder(dtype=tf.float32, shape=[None,28,28,1], name='images')
labels=tf.placeholder(dtype=tf.int32,shape=[None],name='labels')

def model(images,labels):
    
    with tf.name_scope('conv1') as scope:
        weights=tf.Variable(tf.truncated_normal([5,5,1,16], stddev=0.01), name='w')
        bias=tf.Variable(tf.constant(0.1, shape=[16]),name='bias')
        conv1=tf.nn.conv2d(images, weights, strides=[1,2,2,1], padding='SAME', name='conv')
        conv1=tf.nn.bias_add(conv1, bias)
        conv1=tf.nn.relu(conv1, name='relu')
        
    with tf.name_scope('conv2'):
        weights=tf.Variable(tf.truncated_normal([3,3,16,32],stddev=0.01),name='w')
        bias=tf.Variable(tf.constant(0.01,shape=(32,)),name='bias')
        conv2=tf.nn.conv2d(conv1,weights,strides=[1,1,1,1],padding='SAME',name='conv')
        conv2=tf.nn.bias_add(conv2, bias)
        conv2=tf.nn.relu(conv2)
        pool2=tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool')
   
    with tf.name_scope('conv3'):
        weights=tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.01),name='w')
        bias=tf.Variable(tf.constant(0.1,shape=(64,)),name='bias')
        conv3=tf.nn.conv2d(pool2,weights,strides=[1,1,1,1],padding='SAME',name='conv')
        conv3=tf.nn.bias_add(conv3,bias)
        conv3=tf.nn.relu(conv3)
        pool3=tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name='max_pool')
        
    with tf.name_scope('conv4'):
        weights=tf.Variable(tf.truncated_normal([3,3,64,128],stddev=0.01),name='w')
        bias=tf.Variable(tf.constant(0.1,shape=(128,)),name='bias')
        flattened=tf.nn.conv2d(pool3,weights,strides=[1,1,1,1],padding='VALID')
        flattened=tf.nn.bias_add(flattened,bias)
        flattened=tf.nn.relu(flattened)
        flattened=tf.squeeze(flattened, axis=(1,2), name='squeeze')
    
    with tf.name_scope('logits'):
        weights=tf.Variable(tf.truncated_normal(shape=[128,10],stddev=sqrt(2.0/138)),name='w')
        bias=tf.Variable(tf.constant(0.01,shape=(10,)),name='bias')
        logits=tf.matmul(flattened, weights, name='fc')
        logits=tf.add(logits, bias)
        
    with tf.name_scope('loss'):
        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='loss')
        loss=tf.reduce_mean(loss)
        tf.summary.scalar('loss', loss)
        
    with tf.name_scope('predictions'):
        predicts=tf.arg_max(logits, dimension=-1, name='prediction')
    
    return predicts,loss

predicts,loss=model(images,labels)
train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

sess=tf.Session()
writer=tf.summary.FileWriter('./logs/fashion-summary/',sess.graph)
summary_merge=tf.summary.merge_all()
saver=tf.train.Saver()

sess.run(tf.global_variables_initializer())
num_batches=60000//32+1
cur_step=1
for i in range(epochs):
    for j in range(num_batches):
        train_images,train_labels=data.train.next_batch(batch_size)
   
        train_images=np.reshape(train_images, newshape=(batch_size,28,28,1))
    
        #print(train_labels.shape)
        #train_images=tf.reshape(train_images, [None,28,28,1])
        _,loss_val=sess.run([train_op,loss],feed_dict={images:train_images,labels:train_labels})
        if j % 10 == 0:
            print('epoch-%d-step-%d:%.4f' % (i,j,loss_val))
            merged=sess.run(summary_merge,feed_dict={images:train_images,labels:train_labels})
            writer.add_summary(merged, i*j+j)
        
        
    
    saver.save(sess, save_path=save_path, global_step=global_step)
       