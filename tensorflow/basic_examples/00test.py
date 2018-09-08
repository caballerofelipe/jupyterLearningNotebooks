# To avoid warnings:
# See https://stackoverflow.com/a/48486720/1071459
import warnings
warnings.filterwarnings('ignore', '.*Conversion of the second argument.*',)

# To avoid tensorflow warnings
# See https://stackoverflow.com/a/47227886/1071459
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

w = tf.Variable(0,dtype=tf.float32) # Creates warning in current setup 2018-05-18 see in header: https://stackoverflow.com/a/48486720/1071459
cost = tf.add(tf.add(w**2,tf.multiply(-10.,w)),25) # same as cost = w**2 + -10*w + 25
# Don't know why Andrew uses this form, maybe just to show that can be done
#  and probably some operations need to be used like that
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session() # Creates TF warning in current setup 2018-05-18 see in header: https://stackoverflow.com/a/48486720/1071459
session.run(init)
print(session.run(w))

# One iteration
session.run(train)
print('%s%6i %s' % ('Iterations: ', 1, ', w value: '), session.run(w))

iterations = 1000
for i in range(iterations):
    session.run(train)
print('%s%6i %s' % ('Iterations: ', iterations, ', w value: '), session.run(w))
