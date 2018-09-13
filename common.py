# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 22:25:20 2018

@author: Jhon GIl Sepulveda
"""

# In[1]:
import tensorflow as tf

# In[2]:
t1 = tf.constant([1.5, 23, 1.6])
with tf.Session() as sess:
    print(sess.run(t1))
    
# In[3]:
t2 = tf.constant([['c', 'c'], ['c', 'c']])
with tf.Session() as sess:
    print(sess.run(t2))
    
# In[4]:
t3 = tf.constant([4,2], tf.int16, [3], 'Const', True)
with tf.Session() as sess:
    print(sess.run(t3))
    
# In[5]:
zero_tensor = tf.zeros([3])
with tf.Session() as sess:
    print(sess.run(zero_tensor))
    
# In[6]:
fill = tf.fill([2,3], 5)
with tf.Session() as sess:
    print(sess.run(fill))
    
# In[7]:
lin_tensor = tf.lin_space(5., 9., 5)
with tf.Session() as sess:
    print(sess.run(lin_tensor))
    
# In[8]:
rage = tf.range(3, 18, 3)
with tf.Session() as sess:
    print(sess.run(rage))
    
# In[9]:
valores_aleatorios = tf.random_normal([10], dtype=tf.float64)
with tf.Session() as sess:
    print(sess.run(valores_aleatorios))
    
# In[10]:
vector = tf.constant([1.,2.,3.,4.])
matriz = tf.reshape(vector, [2,2])
with tf.Session() as sess:
    print(sess.run(matriz))
    
# In[11]:
matriz = tf.constant([[1.,2.,3.], [4.,5.,6.], [7.,8.,9.]])
slice_matriz = tf.slice(matriz, [1,1], [2,2])
with tf.Session() as sess:
    print(sess.run(slice_matriz))
    
# In[12]:
