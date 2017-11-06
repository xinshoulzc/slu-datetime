import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np

if __name__ == "__main__":
    x = tf.get_variable("X", [2, 3])
    y = tf.get_variable("Y", [2, 3])
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print (sess.run(tf.stack([x,y])))
        print ("==============")
        print (sess.run(y))

