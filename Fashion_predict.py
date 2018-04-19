# --*-- coding:utf-8 --*--
'''
Created on 2018年4月18日

@author: Administrator
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


meta_graph_name='./logs/fashion-model.ckpt-19.meta'
mode_name='./logs/fashion-model.ckpt-19'

sess=tf.Session()
meta_graph=tf.train.import_meta_graph(meta_graph_name)
meta_graph.restore(sess, mode_name)
images=sess.graph.get_tensor_by_name('images:0')
labels=sess.graph.get_tensor_by_name('labels:0')
predictions=sess.graph.get_tensor_by_name('predictions/prediction:0')

batch_size=500
num_batches=10000//500
accuracy=0
data=input_data.read_data_sets('data/fashion')
for i in range(num_batches):
    train_images,train_labels=data.test.next_batch(batch_size)
    train_images=np.reshape(train_images, (batch_size,28,28,1))
    predits=sess.run(predictions,feed_dict={images:train_images})
    accuracy+=np.sum(predits==train_labels)

print('Accuracy: %.3f' % (accuracy/10000.0))
sess.close()