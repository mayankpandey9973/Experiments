# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile
import pdb
import random as rand

import numpy as np
from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

SCALE = 0.001
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/mayankp/tmp/cifar10_dataL2-' + str(SCALE),
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.0     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 128.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
WEIGHT_DECAY = 0.0001

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  # var = _variable_on_cpu(name, shape,
  #                        tf.truncated_normal_initializer(stddev=stddev))
  var = _variable_on_cpu(name, None,
    tf.random_normal(shape) * tf.sqrt(1. / (shape[0]*shape[1]*shape[2])))

  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _variable_with_id_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  # var = _variable_on_cpu(name, shape,
  #                        tf.truncated_normal_initializer(stddev=stddev))
  init_val = tf.random_normal(shape) * tf.sqrt(
      1. / (shape[0]*shape[1]*shape[2]))
  if shape[2] == shape[3]:
    I = create_identity_filter(shape)
  else:
    I = create_identity_filter(shape)

  # alpha1 = 1. / np.sqrt(2.)
  # alpha2 = 1. / np.sqrt(2.)
  alpha1 = 1.
  alpha2 = 1.
  # var = _variable_on_cpu(name, None, I*alpha1+init_val*alpha2)
  var = _variable_on_cpu(name, None, init_val)

  if wd is not None:
    
    U = (var - alpha1 * I) / alpha2
    weight_decay = tf.mul(tf.nn.l2_loss(U), wd, name='weight_loss')
    
    tf.add_to_collection('losses', weight_decay)
  return var

def create_identity_filter(shape):
  filter = np.zeros(shape)
  offset = int((shape[0] - 1) / 2)
  for i in xrange(shape[2]):
    filter[offset, offset, i, i] = 1.
  return filter

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  return cifar10_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)

def standard_distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  return cifar10_input.standard_distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)

def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=FLAGS.batch_size)

def ram_inputs(unit_variance, is_train):
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  return cifar10_input.ram_inputs(data_dir=FLAGS.data_dir,
      unit_variance=unit_variance, is_train=is_train)

def inference(images, n, use_batchnorm, use_nrelu, id_decay, add_shortcuts,
    is_train):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #

  # Weight decay
  weight_decay = WEIGHT_DECAY
  group_shapes = [32, 32, 64, 128]

  # Initial conv block
  with tf.variable_scope('conv0') as scope:
    conv0  = convblock(images, [3, 3, 3, group_shapes[0]], 0, weight_decay,
      use_batchnorm, use_nrelu, id_decay, is_train)
    _activation_summary(conv0)
  
  # grp1
  res1 = addgroup(1, conv0, group_shapes, weight_decay, is_train, n, True,
      use_batchnorm, use_nrelu, id_decay, add_shortcuts)

  # pool1
  pool1 = tf.nn.avg_pool(res1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  
  # grp2
  res2 = addgroup(2, pool1, group_shapes, weight_decay, is_train, n, False,
      use_batchnorm, use_nrelu, id_decay, add_shortcuts)

  # pool2
  pool2 = tf.nn.avg_pool(res2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # grp3
  res3 = addgroup(3, pool2, group_shapes, weight_decay, is_train, n, False,
      use_batchnorm, use_nrelu, id_decay, add_shortcuts)

  # logit
  with tf.variable_scope('softmax_linear') as scope:
    
    relu_bias = 0.399
    relu_std = 0.584
    # Add bnorm and relu after the last grp.
    groups_out = res3
    if use_batchnorm:
      groups_out = batchnorm(groups_out, '1', is_train)
    if use_nrelu:
      groups_out = (tf.nn.relu(groups_out) - relu_bias) / relu_std
    else:
      groups_out = tf.nn.relu(groups_out)
    
    kernel = _variable_with_weight_decay('weights',
        shape=[1, 1, group_shapes[3], NUM_CLASSES], stddev=1e-4, 
        wd=weight_decay)
    conv = tf.nn.conv2d(groups_out, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [NUM_CLASSES], 
      tf.constant_initializer(0.0))
    convlogit = tf.nn.bias_add(conv, biases)
    
    # avg pool the logits
    dim = convlogit.get_shape()[2].value
    softmax_linear = tf.nn.avg_pool(convlogit, ksize=[1, dim, dim, 1],
      strides=[1, 1, 1, 1], padding='VALID', name='avg_pool')
    softmax_linear = tf.reshape(softmax_linear, [FLAGS.batch_size, -1])
    _activation_summary(softmax_linear)

  return softmax_linear

def addgroup(grp_id, input, group_shapes, weight_decay, is_train, num_blocks,
    is_first_group, use_batchnorm, use_nrelu, id_decay, add_shortcuts):
  
  with tf.variable_scope('grp'+str(grp_id)):
    res = input
    for k in range(0, num_blocks):
      dont_add_relu = False
      if k == 0:
        shape = [3, 3, group_shapes[grp_id-1], group_shapes[grp_id]]
        if is_first_group:
          dont_add_relu = True
      else:
        shape = [3, 3, group_shapes[grp_id], group_shapes[grp_id]]

      with tf.variable_scope('res'+str(k)) as scope:
        res  = residualblock(res, shape, k, dont_add_relu, weight_decay,
            use_batchnorm, use_nrelu, id_decay, add_shortcuts, is_train)
    _activation_summary(res)
  return res

def grpLoss(grp_id, scale):
    mu0 = tf.reduce_mean(tf.get_collection('wts0', 'grp' + str(grp_id)), 0)
    mu1 = tf.reduce_mean(tf.get_collection('wts1', 'grp' + str(grp_id)), 0)
    return scale * (tf.nn.l2_loss(tf.get_collection('wts0', 'grp' + str(grp_id)) - mu0) + 
	    tf.nn.l2_loss(tf.get_collection('wts1', 'grp' + str(grp_id)) - mu1))

def batchnorm(input, suffix, is_train):
  rank = len(input.get_shape().as_list())
  in_dim = input.get_shape().as_list()[-1]

  if rank == 2:
      axes = [0]
  elif rank == 4:
      axes = [0, 1, 2]
  else:
      raise ValueError('Input tensor must have rank 2 or 4.')

  mean, variance = tf.nn.moments(input, axes)
  offset = _variable_on_cpu('offset_' + str(suffix), in_dim,
      tf.constant_initializer(0.0))
  scale = _variable_on_cpu('scale_' + str(suffix), in_dim, 
    tf.constant_initializer(1.0))
  # offset = 0.0
  #scale = 1.0
  
  decay = 0.95
  epsilon = 1e-4

  ema = tf.train.ExponentialMovingAverage(decay=decay)
  ema_apply_op = ema.apply([mean, variance])
  ema_mean, ema_var = ema.average(mean), ema.average(variance)

  if is_train:
      with tf.control_dependencies([ema_apply_op]):
          return tf.nn.batch_normalization(
              input, mean, variance, offset, scale, epsilon)
  else:
      # batch = tf.cast(tf.shape(x)[0], tf.float32)
      # mean, var = ema_mean, ema_var * batch / (batch - 1) # unbiased variance estimator
      return tf.nn.batch_normalization(
          input, ema_mean, ema_var, offset, scale, epsilon)

def residualblock(input, shape, suffix, first, weight_decay, use_batchnorm,
    use_nrelu, id_decay, add_shortcuts, is_train):
  # Do the projection as well.

  relu_bias = 0.399
  relu_std = 0.584

  shortcut = input
  kernel_ = [0, 0]
  if shape[2] != shape[3]:
    # Do projection.
    proj_shape = [1, 1, shape[2], shape[3]]
    wt_name = 'weights_proj_' + str(suffix)
    if id_decay:
      kernel = _variable_with_id_decay(wt_name, shape=proj_shape,
                                       stddev=1e-4, wd=weight_decay)
    else:
      kernel = _variable_with_weight_decay(wt_name, shape=proj_shape,
                                           stddev=1e-4, wd=weight_decay)
    shortcut = tf.nn.conv2d(shortcut, kernel, [1, 1, 1, 1], padding='SAME')

  if not first:
    if use_batchnorm:
      input = batchnorm(input, '1_' + str(suffix), is_train)
    if use_nrelu:
      input = (tf.nn.relu(input) - relu_bias) / relu_std
    else:
      input = tf.nn.relu(input)
  wt_name = 'weights_1_' + str(suffix)
  if id_decay:
    kernel_[0] = _variable_with_id_decay(wt_name, shape=shape,
                                     stddev=1e-4, wd=weight_decay)
  else:
    kernel_[0] = _variable_with_weight_decay(wt_name, shape=shape,
                                         stddev=1e-4, wd=weight_decay)
  if shape[2] == shape[3]: 
    tf.add_to_collection('wts0', kernel_[0])
  conv = tf.nn.conv2d(input, kernel_[0], [1, 1, 1, 1], padding='SAME')
  b_name = 'biases_1_' + str(suffix)
  biases = _variable_on_cpu(b_name, shape[3], tf.constant_initializer(0.0))
  bias = tf.nn.bias_add(conv, biases)
  #bias = conv

  # Do batch norm as well
  if use_batchnorm:
    input = batchnorm(input, '2_' + str(suffix), is_train)
  if use_nrelu:
    input = (tf.nn.relu(bias) - relu_bias) / relu_std
  else:
    input = tf.nn.relu(bias)
  
  # Upsampling (if needed) happens in the first conv block above.
  shape[2] = shape[3]
  wt_name = 'weights_2_' + str(suffix)
  if id_decay:
    kernel_[1] = _variable_with_id_decay(wt_name, shape=shape,
                                         stddev=1e-4, wd=weight_decay)
  else:
    kernel_[1] = _variable_with_weight_decay(wt_name, shape=shape,
                                           stddev=1e-4, wd=weight_decay)
  
  conv = tf.nn.conv2d(input, kernel_[1], [1, 1, 1, 1], padding='SAME')
  tf.add_to_collection('wts1', kernel_[1])
  b_name = 'biases_2_' + str(suffix)
  biases = _variable_on_cpu(b_name, shape[3], tf.constant_initializer(0.0))
  bias = tf.nn.bias_add(conv, biases)
  #bias = conv

  if False:
    return 0

  # Add the x
  if add_shortcuts:
    res = bias + shortcut
  else:
    res = bias

  return res

def convblock(input, shape, suffix, weight_decay, use_batchnorm, 
    use_nrelu, id_decay, is_train):
  wt_name = 'weights' + str(suffix)
  if id_decay:
    kernel = _variable_with_id_decay(wt_name, shape=shape,
                                     stddev=1e-4, wd=weight_decay)
  else:
    kernel = _variable_with_weight_decay(wt_name, shape=shape,
                                         stddev=1e-4, wd=weight_decay)
  conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
  b_name = 'biases' + str(suffix)
  biases = _variable_on_cpu(b_name, shape[3], tf.constant_initializer(0.0))
  bias = tf.nn.bias_add(conv, biases)
  #bias = conv

  relu_bias = 0.399
  relu_std = 0.584

  conv1 = bias
  if use_batchnorm:
    conv1 = batchnorm(conv1, suffix, is_train)
  if use_nrelu:
    conv1 = (tf.nn.relu(conv1) - relu_bias) / relu_std
  else:
    conv1 = tf.nn.relu(conv1)

  return conv1

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  tf.add_to_collection('losses', grpLoss(1, SCALE))
  tf.add_to_collection('losses', grpLoss(2, SCALE))
  tf.add_to_collection('losses', grpLoss(3, SCALE))

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  print('num batch per epoch', num_batches_per_epoch, 'decay steps', decay_steps)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.MomentumOptimizer(lr, 0.9)
    # opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def get_max_steps(is_train, num_epochs):
  if is_train:
    max_steps = (cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN 
        * num_epochs) / float(FLAGS.batch_size)
  else:
    max_steps = (cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        * num_epochs) / float(FLAGS.batch_size)
  
  return int(np.ceil(max_steps))


class Cifar10BatchGenerator(object):
  """Batch generator for Cifar10.

  TODO:
  1) Add random cropping.
  2) Add random LR flip.
  """

  def __init__(self, images, labels, is_train, num_epochs):
    self.images = images
    self.labels = labels
    self.is_train = is_train
    self.batch_size = FLAGS.batch_size
    self.num_epochs = num_epochs
    self.curr_ind = 0
    self.curr_epoch = 0
    self._padding = 4
    self.done = False
    self._num_samples = images.shape[0]
    # Assume square image.
    self._imsize = images.shape[1]

    if is_train:
      self.shuffle()
      # Pad for random cropping.
      padding = [[0, 0], [4, 4], [4, 4], [0, 0]]
      self.images = np.pad(self.images, padding, 'constant')

  def is_done(self):
    return self.done

  def num_samples(self):
    return self._num_samples

  def shuffle(self):
    perm = np.arange(self.images.shape[0])
    np.random.shuffle(perm)
    self.images = self.images[perm]
    self.labels = self.labels[perm]

  def next_batch(self):
    end_ind = self.curr_ind + self.batch_size
    do_shuffle = False
    if end_ind >= self.images.shape[0]:
      # Epoch finished
      end_ind = self.images.shape[0]
      self.curr_epoch = self.curr_epoch + 1
      if self.curr_epoch == self.num_epochs:
        self.done = True
      # Shuffle the dataset
      do_shuffle = True

    indices = slice(self.curr_ind, end_ind)
    images = self.images[indices]
    labels = self.labels[indices]

    if end_ind == self.images.shape[0]:
      self.curr_ind = 0
    else:
      self.curr_ind = end_ind

    # Do the augmentation
    if self.is_train:
      # Random crop
      offset = np.random.randint(self._padding*2+1,size=2)
      r = slice(offset[0],offset[0]+self._imsize)
      c = slice(offset[1],offset[1]+self._imsize)
      images = images[:,r,c,:]
      # LR flip.
      if np.random.rand(1) > 0.5:
        images = images[:,:,::-1,:]

    if do_shuffle:
      self.shuffle()

    return images, labels
 

