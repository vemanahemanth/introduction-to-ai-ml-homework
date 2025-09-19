import tensorflow as tf

X = tf.constant([1., 2., 3., 4., 5.])
Y = tf.constant([5., 8., 11., 14., 17.])

m = tf.reduce_mean((X - tf.reduce_mean(X)) * (Y - tf.reduce_mean(Y))) / tf.math.reduce_variance(X)
c = tf.reduce_mean(Y) - m * tf.reduce_mean(X)

prediction = m * 6 + c

print("Slope (m):", m.numpy())
print("Intercept (c):", c.numpy())
print("Prediction for x=6:", prediction.numpy())
