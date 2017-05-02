#TensorFlow Machine Learning Cookbook
#coding:utf8

import cv2
import numpy as np
img = cv2.imread("./02.jpg")
print img
cv2.imshow("inputaa", img)
#height, width = img.shape[:2]

import tensorflow as tf
def pp(v):
	print v.eval() 
	
sess	=	tf.InteractiveSession()	
row_dim, col_dim = 5, 4
#创建tensorflow
#3行2列，全0
zero_tsr = tf.zeros([3, 2])
print zero_tsr.eval()
#3行2列，全1
ones_tsr = tf.ones([row_dim, col_dim])
print ones_tsr.eval()
#3行2列，全42
filled_tsr = tf.fill([row_dim, col_dim], 42)
print filled_tsr.eval()
#指定的常量
constant_tsr = tf.constant([[1,2,3],[4, 0.5, 6],[11, 34, 5.6]])
print constant_tsr.eval()

#利用已有tensor，生成新的同型的tensor
zeros_similar = tf.zeros_like(constant_tsr)
#ones_similar = tf.ones_like(constant_tsr)
ones_similar = tf.ones_like(ones_tsr)
print zeros_similar.eval()
print ones_similar.eval()
pp(zeros_similar)

#创建数列tensor : Sequence tensors
linear_tsr = tf.linspace(start=1.0, stop=100.0, num = 20)
integer_seq_tsr = tf.range(start=6, limit=15, delta=3)
print "*"*20
pp(linear_tsr)
pp(integer_seq_tsr)

print "*"*20
#随机数据作为tensor值
randunif_tsr = tf.random_uniform([row_dim, col_dim], minval=0, maxval=1)
pp(randunif_tsr)
#满足正态分布的随机数据
randnorm_tsr = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)
pp(randnorm_tsr)
#stddev是方差，数据也是满足正态分布的
runcnorm_tsr = tf.truncated_normal([row_dim, col_dim],mean=0.0, stddev=1.0)
pp(runcnorm_tsr)
print "*"*20
#Randomly shuffles a tensor along its first dimension.
#shuffle 随机
shuffled_output = tf.random_shuffle(randnorm_tsr)
pp(randnorm_tsr)
pp(shuffled_output)
shuffled_output = tf.random_shuffle(constant_tsr)
pp(constant_tsr)
pp(shuffled_output)
print "read a img" * 5
#'''


height, width = img.shape[:2]
print height,width
for x in range(12):
	cropped_image = tf.random_crop(img, (height / 4, width / 4,3))
#pp(cropped_image)
	cv2.imshow("output" + str(x), cropped_image.eval())
cv2.waitKey(10) 

#'''

#pp(sess.run(my_var))
'''
Variables are the parameters of the algorithm and TensorFlow keeps track of how
to change these to optimize the algorithm. 

Placeholders are objects that allow you to feed in
data of a specific type and shape and depend on the results of the computational graph, such
as the expected outcome of a computation.
'''
print "-"*20
# This is the declaration and we still need to initialize the variable
my_var = tf.Variable(constant_tsr)
pp(constant_tsr)
#Placeholders are just holding the position for data to be fed into the graph
x = tf.placeholder(tf.float32, shape=[2,2])
y = tf.identity(x)
x_vals = np.random.rand(2,2)

# initialize variable
initialize_op = tf.initialize_all_variables()
sess1 = tf.Session()
sess1.run(initialize_op)
print "+"*20
#Placeholders get data from a feed_dict argument in the session. 
print sess1.run(y, feed_dict={x: x_vals})
print sess1.run(my_var)
sess1.close()

sess2 = tf.Session()
first_var = tf.Variable(tf.zeros([2,3]))
# each variable has an initializer method
print sess2.run(first_var.initializer)
second_var = tf.Variable(tf.zeros_like(first_var))
# Depends on first_var
print sess2.run(second_var.initializer)
sess2.close()





























sess.close()
