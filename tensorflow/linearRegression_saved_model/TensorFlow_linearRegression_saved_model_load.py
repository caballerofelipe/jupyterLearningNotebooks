# Prepare stuff
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os  # Used to obtain this file's current path

# See https://stackoverflow.com/a/47227886/1071459
# Just disables the warning, doesn't enable AVX/FMA
# To avoid tensorflow warnings, uncomment the following line
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

# Config
num_epochs = 10000

# For plotting
xrange = np.linspace(-10, 30, 2)

with tf.Session() as sess:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],  # ["serve"]
        f"{dir_path}/saved_model/"
    )

    costTensor = sess.graph.get_tensor_by_name("cost:0")
    optimizerOperation = sess.graph.get_operation_by_name("optimizer")
    a = sess.graph.get_tensor_by_name("a:0")
    b = sess.graph.get_tensor_by_name("b:0")

    costEval = costTensor.eval(feed_dict={
        "X:0": x,
        "Y:0": y
    })
    loaded_a = a.eval()
    loaded_b = b.eval()
    print("After loading")
    print("cost[{:9.4f}] / a[{:7.4f}] / b[{:7.4f}]".format(
        costEval, loaded_a, loaded_b)
    )
    print("Inference for some values:")
    x_test_val = [0, 1, 2]
    h_values = sess.run("h:0", feed_dict={"X:0": x_test_val})  # Inference
    for i in range(len(x_test_val)):
        print("For x={} the predicted value is: {:.2f} (rounded to two decimals).".format(
            x_test_val[i], h_values[i]
        ))

    # For plotting
    loaded_yrange = sess.run("h:0", feed_dict={"X:0": xrange})  # Inference

    print("\nTrain {} more epochs".format(num_epochs))
    for epoch in range(num_epochs):
        _, costValue = sess.run(
            [optimizerOperation, costTensor],
            feed_dict={
                "X:0": x,
                "Y:0": y
            }
        )
        costEval = costTensor.eval(feed_dict={
            "X:0": x,
            "Y:0": y
        })
        new_a = a.eval()
        new_b = b.eval()
        if epoch % (num_epochs / 10) == 0 or epoch == num_epochs - 1:  # Every 10 percent and Last epoch
            print("... epoch: {:4}".format(epoch), end=", ")
            print("cost[{:9.4f}] / a[{:7.4f}] / b[{:7.4f}]".format(
                costEval, new_a, new_b)
            )

    print("\nAfter some training")
    print("cost[{:9.4f}] / a[{:7.4f}] / b[{:7.4f}]".format(
        costEval, new_a, new_b)
    )
    print("Inference for some values:")
    x_test_val = [0, 1, 2]
    h_values = sess.run("h:0", feed_dict={"X:0": x_test_val})  # Inference
    for i in range(len(x_test_val)):
        print("For x={} the predicted value is: {:.2f} (rounded to two decimals).".format(
            x_test_val[i], h_values[i]
        ))

    # For plotting
    new_yrange = sess.run("h:0", feed_dict={"X:0": xrange})  # Inference

# Seing the obtained values in a plot
xrange = np.linspace(-10, 30, 2)
# Plot points
plt.plot(x, y, "ro")
# Plot loaded function
plt.plot(xrange, loaded_yrange, "b", label="loaded")
plt.plot(xrange, new_yrange, "c", label="new")
plt.legend()
plt.show()
