import tensorflow as tf
import time
import numpy as np
import pandas as pd

# data = pd.DataFrame(np.load("train.npy"))
tf.logging.set_verbosity(tf.logging.INFO)

base_lr = 1e-5
momentum = 0.99
power = 0.75
decay_rate = 0.0001
wd = 0.005
mi = 10000
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

lenet_graph = tf.Graph()
sess1 = tf.InteractiveSession(graph=lenet_graph)

with lenet_graph.as_default():
  global_step = tf.Variable(0, trainable=False)

  learning_rate = tf.train.inverse_time_decay(base_lr,global_step,decay_steps,decay_rate)

  x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
  y_ = tf.placeholder(tf.float32, shape=[None, 2])

  W_conv1 = weight_variable([5,5,3,20])
  b_conv1 = bias_variable([20])

  x_image = x
  h_conv1 = conv2d(x_image, W_conv1) + b_conv1
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 20, 50])
  b_conv2 = bias_variable([50])

  h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([8 * 8 * 50, 500])
  b_fc1 = bias_variable([500])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 50])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([500, 2])
  b_fc2 = bias_variable([2])

  y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

  reg = wd * (tf.nn.l2_loss(W_conv1) 
                  + tf.nn.l2_loss(W_conv2) 
                  + tf.nn.l2_loss(W_fc1) 
                  + tf.nn.l2_loss(W_fc2))

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv) + reg)
      
  optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum)
  train_step = optimizer.minimize(cross_entropy,global_step=global_step)

  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

  prediction = tf.argmax(y_conv,1)

  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # sess1.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  saver.restore(sess1, './lenet_model/')



def train(data):
  start = time.time()
  for i in range(mi):
    batch = data.sample(100)
    X = np.stack(batch[0].values)
    Y = np.stack(batch[1].values)
    if i%100 == 0:
      train_accuracy = accuracy.eval(session=sess1,feed_dict={x:np.float32(X), y_: Y, keep_prob: 1.0})
      print("step %d, training accuracy %g%%"%(sess1.run(global_step), train_accuracy*100))
      print("loss: ",cross_entropy.eval(session=sess1,feed_dict={x:np.float32(X), y_: Y, keep_prob: 1.0}))
    train_step.run(feed_dict={x: X, y_: Y, keep_prob: 0.5})

  print("Time Elapsed ",time.time() - start)
  saver.save(sess1, './lenet_model/')

def test(data):
  test_accuracy = 0
  for i in range(200):
    batch = data[i*500:(i+1)*500]
    X = np.stack(batch[0].values)
    Y = np.stack(batch[1].values)
    test_accuracy += accuracy.eval(session=sess1,feed_dict={x:np.float32(X), y_: Y, keep_prob: 1.0})
  print("Test Accuracy : %.2f%%" % (test_accuracy/2))

def predict(images):
  return prediction.eval(session=sess1,feed_dict={x:np.float32(images), keep_prob: 1.0})

def confidence(images):
  return y_conv.eval(session=sess1,feed_dict={x:np.float32(images), keep_prob: 1.0})
