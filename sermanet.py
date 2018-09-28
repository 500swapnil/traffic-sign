import tensorflow as tf
import time
import numpy as np
import pandas as pd

# data = pd.DataFrame(np.load("train.npy"))
tf.logging.set_verbosity(tf.logging.INFO)

""" Classification Dictionary
    
    0:'Advance Left'
    1:'Advance Right'
    2:'Breaker'
    3:'Compulsory Ahead'  
    4:'Compulsory Left'
    5:'Compulsory Right'
    6:'Stop'
    7:'Traffic Signal'

"""

base_lr = 1e-4
momentum = 0.9
power = 0.75
decay_rate = 0.0001
wd = 0.005
mi = 6000
decay_steps = 100

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sermanet_graph = tf.Graph()
sess = tf.InteractiveSession(graph=sermanet_graph)

with sermanet_graph.as_default():
  x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
  y_ = tf.placeholder(tf.float32, shape=[None, 8])

  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.inverse_time_decay(base_lr,global_step,decay_steps,decay_rate)
  
  W_conv1 = weight_variable([5,5,3,32])
  b_conv1 = bias_variable([32])

  x_image = x
  conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  pool1 = max_pool_2x2(conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  conv2 = tf.nn.relu(conv2d(pool1, W_conv2) + b_conv2)
  pool2 = max_pool_2x2(conv2)

  W_conv3 = weight_variable([5, 5, 64, 128])
  b_conv3 = bias_variable([128])

  conv3 = tf.nn.relu(conv2d(pool2, W_conv3) + b_conv3)
  pool3 = max_pool_2x2(conv3)

  pool4 = tf.nn.max_pool(pool1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

  pool5 = max_pool_2x2(pool2)

  pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 128])
  pool4_flat = tf.reshape(pool4, [-1, 4 * 4 * 32])
  pool5_flat = tf.reshape(pool5, [-1, 4 * 4 * 64])

  concat = tf.concat([pool3_flat, pool4_flat, pool5_flat], axis=1)

  W_fc1 = weight_variable([4 * 4 *(128 + 32 + 64), 1024])
  b_fc1 = bias_variable([1024])
  fc1 = tf.nn.relu(tf.matmul(concat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(fc1, keep_prob)

  W_fc2 = weight_variable([1024, 8])
  b_fc2 = bias_variable([8])

  logits = tf.matmul(fc1, W_fc2) + b_fc2

  reg = wd * (tf.nn.l2_loss(W_conv1)
                  + tf.nn.l2_loss(W_conv2)
                  + tf.nn.l2_loss(W_conv3)
                  + tf.nn.l2_loss(W_fc1)
                  + tf.nn.l2_loss(W_fc2))

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits) + reg)

  optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum)
  train_step = optimizer.minimize(cross_entropy,global_step=global_step)

  correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))

  prediction = tf.argmax(logits,1)

  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  saver = tf.train.Saver()

  # sess.run(tf.global_variables_initializer())
  saver.restore(sess, './sermanet_model/')

def train(data):
  start = time.time()
  for i in range(mi):
    batch = data.sample(100)
    X = np.stack(batch[0].values)
    Y = np.stack(batch[1].values)
    if i%100 == 0:
      train_accuracy = accuracy.eval(session=sess,feed_dict={x:np.float32(X), y_: Y, keep_prob: 1.0})
      print("step %d, training accuracy %g%%"%(sess.run(global_step), train_accuracy*100))
      print("loss: ",cross_entropy.eval(session=sess,feed_dict={x:np.float32(X), y_: Y, keep_prob: 1.0}))
    train_step.run(feed_dict={x: X, y_: Y, keep_prob: 0.5})

  print("Time Elapsed ",time.time() - start)
  saver.save(sess, './sermanet_model/')

def test(data):
  test_accuracy = 0
  for i in range(200):
    batch = data[i*500:(i+1)*500]
    X = np.stack(batch[0].values)
    Y = np.stack(batch[1].values)
    test_accuracy += accuracy.eval(session=sess,feed_dict={x:np.float32(X), y_: Y, keep_prob: 1.0})
  print("Test Accuracy : %.2f%%" % (test_accuracy/2))

def predict(images):
  return prediction.eval(session=sess,feed_dict={x:np.float32(images), keep_prob: 1.0})
