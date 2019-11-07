import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

with tf.device('/gpu:0'):
    v1 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v1')
    v2 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v2')
    sumV = v1 + v2

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    try:
        print(sess.run(sumV))
        print("Success to run with gpu")
    except BaseException as e:
        print("Failed to run with gpu")
        print(e)