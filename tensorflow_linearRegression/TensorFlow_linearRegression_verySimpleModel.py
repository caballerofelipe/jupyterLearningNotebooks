# Prepare stuff
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

# Prepare the data
x = np.array([
    -7,
    5,
    7,
    12,
    2,
    4,
    5,
    6,
    19,
    14,
    16,
    15,
    11,
    8,
    18,
    -2,
    14,
    17,
    7,
    17,
])
y = np.array([
    -7.56,
    21.01,
    29.11,
    47.89,
    14.11,
    16.69,
    29.81,
    28.67,
    63.94,
    54.8,
    49.65,
    52.26,
    44.86,
    40.45,
    70.92,
    -0.37,
    44.34,
    64.27,
    32.85,
    50.14,
])

# Check the data
print('x.shape():')
print(x.shape)
print('y.shape():')
print(y.shape)
plt.scatter(x, y)
plt.show()

# TensorFlow Model

# Config
num_epochs = 1000
learning_rate = 0.001
# /Config

# Creating the graph
ops.reset_default_graph()

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

a = tf.get_variable('a', initializer=0.)
b = tf.get_variable('b', initializer=0.)

h = a * X + b

cost = tf.reduce_mean( (h - Y)**2 )

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate
).minimize(cost)

init = tf.global_variables_initializer()

# Running the Model
found_a = 0
found_b = 0

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        _, costValue = sess.run(
            [optimizer, cost],
            feed_dict={
                X: x,
                Y: y,
            }
        )
        found_a = a.eval()
        found_b = b.eval()
        if epoch % (num_epochs/10) == 0: # Every 10 percent
            print("... epoch: " + str(epoch))
            print(f"cost[{str(costValue)}] / a[{str(a.eval())}] / b[{str(b.eval())}]")

# Seing the obtained values in a plot
xrange = np.linspace(-10, 30, 2)

# Plot points
plt.plot(x, y, 'ro')

# Plot resulting function
plt.plot(xrange, xrange * found_a + found_b, 'b')

plt.show()
