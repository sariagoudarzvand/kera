import tensorflow as tf
tf.enable_eager_execution()
a = tf.constant([[11,0,13,14],
                 [21,22,23,0]])
condition = tf.equal(a, 0)
print condition
case_true = tf.cast(tf.multiply(tf.ones(shape=(tf.shape(a))), -1000), tf.int32)
print case_true
a_m = tf.where(condition, case_true, a)
print a_m
