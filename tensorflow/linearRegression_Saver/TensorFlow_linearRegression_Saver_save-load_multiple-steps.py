# Prepare stuff
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import os  # Used to obtain this file's current path

# TO BE DOCUMENTED
# import shutil  # WARNING: might not be necessary
# used in conjunction with:
# shutil.rmtree(Saver_model_path)  # Deletes the directory after knowing previous state
# in the else part of print("\nRestoring Variables")

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

tf.train.create_global_step()
# used to keep track of how many saves have happened
recording_count = tf.get_variable("recording_count", initializer=0)

h = a * X + b
tf.identity(h, name='h')

cost = tf.reduce_mean((h - Y)**2, name='cost')

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate
).minimize(cost, name='optimizer', global_step=tf.train.get_global_step())

init = tf.global_variables_initializer()

# Initializing Saver
dir_path = os.path.dirname(os.path.realpath(__file__))
Saver_model_path = dir_path + "/save-load_multiple_steps"
saver = tf.train.Saver(max_to_keep=10)

# Recover last checkpoint
# With the recovered last checkpoint, old saves exceding max_to_keep are deleted
try:
    print("\nRecovering last checkpoint")
    states = tf.train.get_checkpoint_state(Saver_model_path)
    checkpoint_paths = states.all_model_checkpoint_paths
    saver.recover_last_checkpoints(checkpoint_paths)
except AttributeError:
    print(" No saved model found, not recovering previous state.")
else:
    print(" Done")

# Running the Model
found_a = 0
found_b = 0
final_cost = 0

with tf.Session() as sess:
    sess.run(init)

    try:
        print("\nRestoring Variables")
        latest_checkpoint = tf.train.latest_checkpoint(Saver_model_path + "/")
        saver.restore(sess, latest_checkpoint)
    except (
            tf.errors.NotFoundError,
            tf.errors.InvalidArgumentError,
            ValueError
    ):
        print(" Model not found, not restoring.")
    else:
        print(" Restore: Done")
        print(" Latest training step: {}".format(tf.train.get_global_step().eval()))
        print(" recording_count: {}".format(recording_count.eval()))

    print("\nTraining")
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
            print("... epoch(current training)[{}] / global_step[{}]".format(
                epoch, tf.train.get_global_step().eval()
            ))
            print("cost[{:9.4f}] / a[{:7.4f}] / b[{:7.4f}]".format(
                final_cost, found_a, found_b)
            )
            recording_count.assign_add(1).op.run()
            print(" recording_count: {}".format(recording_count.eval()))
            print(" Saving Variables", end="")
            save_path = saver.save(
                sess,
                Saver_model_path + "/model",
                global_step=tf.train.get_global_step())
            print(" Done, saved to\n{}\n".format(save_path))

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
