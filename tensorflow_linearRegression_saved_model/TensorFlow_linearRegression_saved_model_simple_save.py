# Prepare stuff
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import os  # Used to obtain this file's current path

# See https://stackoverflow.com/a/47227886/1071459
# Just disables the warning, doesn't enable AVX/FMA
# To avoid tensorflow warnings, uncomment the following line
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Config
seeplot = False  # Change to True to show plots

# Prepare the data
x = np.array([
    -7, 5, 7, 12, 2,
    4, 5, 6, 19, 14,
    16, 15, 11, 8, 18,
    -2, 14, 17, 7, 17
])
y = np.array([
    -7.56, 21.01, 29.11, 47.89, 14.11,
    16.69, 29.81, 28.67, 63.94, 54.8,
    49.65, 52.26, 44.86, 40.45, 70.92,
    -0.37, 44.34, 64.27, 32.85, 50.14
])

# Check the data
if seeplot:
    plt.scatter(x, y)
    plt.show()

# TensorFlow Model

# Config
num_epochs = 1000
learning_rate = 0.001

# Creating the graph
ops.reset_default_graph()

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

a = tf.get_variable("a", initializer=0.)
b = tf.get_variable("b", initializer=0.)

h = a * X + b
tf.identity(h, name='h')

cost = tf.reduce_mean((h - Y)**2, name='cost')

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate
).minimize(cost, name='optimizer')

init = tf.global_variables_initializer()

# Running the Model
found_a = 0
found_b = 0
final_cost = 0

with tf.Session() as sess:
    sess.run(init)
    # print("\n")
    for epoch in range(num_epochs):
        # Side note:
        #   costValue is the value of cost before the optimizer runs one time
        _, costValue = sess.run(
            [optimizer, cost],
            feed_dict={
                X: x,
                Y: y,
            }
        )
        found_a = a.eval()
        found_b = b.eval()
        final_cost = cost.eval(feed_dict={
            X: x,
            Y: y,
        })
        if epoch % (num_epochs / 10) == 0 or epoch == num_epochs - 1:  # Every 10 percent and Last epoch
            print("... epoch: {:4}".format(epoch), end=', ')
            print("cost[{:9.4f}] / a[{:7.4f}] / b[{:7.4f}]".format(
                final_cost, found_a, found_b)
            )

    # If re-saving, the saved directory must be erased before or it will produce an error
    dir_path = os.path.dirname(os.path.realpath(__file__))
    tf.saved_model.simple_save(
        sess,
        f"{dir_path}/saved_model/",
        inputs={
            "X": X
        },
        outputs={
            "h": h
        }
    )

# Showing results
print("\nThese are the values found during training:")
print(f"a: {found_a}")
print(f"b: {found_b}")

print("\nSome values")
x_test_val = [0, 1, 2]
for x_tmp in x_test_val:
    print("For x={} the predicted value is: {:.2f} (rounded to two decimals).".format(
        x_tmp, x_tmp * found_a + found_b
    ))


if seeplot:
    input("Press enter to continue...")  # Here to avoid shoing the graph before seeing the previous results

    # Seing the obtained values in a plot
    xrange = np.linspace(-10, 30, 2)
    # Plot points
    plt.plot(x, y, 'ro', xrange, xrange * found_a + found_b)
    # Plot resulting function
    plt.plot(xrange, xrange * found_a + found_b, 'b')
    plt.show()
