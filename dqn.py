import tensorflow as tf
import numpy as np
import util

class DQN:

    _id = -1

    def __init__(self, observation_space, action_space,
                batch_size=15, pixels = False):
        DQN._id+=1
        self.id = DQN._id

        if not pixels:
            self._defineModel = self._defineModel_noConv

        with tf.variable_scope('DQN_%d'%(self.id,)):
            self.batch_size = batch_size
            batch_shape = [None,] + list(observation_space.shape)
            self.input = tf.placeholder(dtype=tf.float32, shape=batch_shape, name="input")
            outp = self._defineModel(self.input)
            self.Q_values = tf.layers.dense(inputs=outp, units=action_space.n,
                                        activation=None, name='output')
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32, name="targetQ_ph")
            self.actions = tf.placeholder(dtype=tf.float32, [None], name='actions')
            self.actions_onehot = tf.one_hot(self.actions, action_space.n, axis=1)
            self.Q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions_onehot), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
            self._trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self._trainer.minimize(self.loss)

    def applyUpdate(self, source, tau):
        src_vars = tf.trainable_variables(scope="DQN_%d"%(source.id))
        target_vars = tf.trainable_variables(scope="DQN_%d"%(self.id))
        update_ops = []
        for i in range(len(src_vars)):
            var = util.map_dqn_var(src_vars[i], target_vars)
            op = var.assign(var.value() * (1 - tau) + src_vars[i] * tau)
            update_ops.append(op)
        return update_ops

    def _defineModel(self, inp):
        # inp = tf.reshape(tensor=inp, shape=inp.get_shape().as_list()+[1,])
        conv1 = tf.layers.conv2d(inputs=inp, kernel_size=[1,1],
                                filters=32, activation=tf.nn.relu, name='conv1')
        conv2 = tf.layers.conv2d(inputs=conv1, kernel_size=[1,1],
                                filters=32, activation=tf.nn.relu, name='conv2')
        conv3 = tf.layers.conv2d(inputs=conv2, kernel_size=[1,5],
                                filters=16, activation=tf.nn.relu, name='conv3')
        max_pool = tf.layers.max_pooling2d(conv3, [3,1], [3,1], 'same')
        conv4 = tf.layers.conv2d(inputs=max_pool, kernel_size=[7,1],
                                filters=16, activation=tf.nn.relu, name='conv4')
        max_pool1 = tf.layers.max_pooling2d(conv4, [1,3], [1,3], 'same')
        flat = tf.layers.flatten(max_pool1)
        dense1 = tf.layers.dense(inputs=flat, units=512,
                                activation=tf.nn.relu, name='dense1')
        return dense1

    def _defineModel_noConv(self, inp):
        dense1 = tf.layers.dense(inputs=inp, units=512,
                                activation=tf.nn.relu, name='dense1')
        dense2 = tf.layers.dense(inputs=dense1, units=512,
                                activation=tf.nn.relu, name='dense2')
        return dense2

class experience_buffer:
    def __init__(self, buffer_size=100000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.iter = iter(self.buffer)
        self.i=0

    def add(self, experience):
        if np.array(experience).ndim == 1:
            if len(self.buffer) + 1 >= self.buffer_size:
                self.buffer[0:1] = []
            self.buffer.append(experience)
        else:
            if len(self.buffer) + len(experience) >= self.buffer_size:
                self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
            self.buffer.extend(experience)

    def sample(self,size):
        try:
            if size <= len(self.buffer):
                return np.reshape(np.array(np.random.permutation(self.buffer)[0:size]),[size,4])
            else:
                return np.reshape(np.array(np.random.permutation(self.buffer)[0:size]),[-1,4])
        except ValueError:
            print(size <= len(self.buffer))
            raise
