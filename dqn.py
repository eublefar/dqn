import tensorflow as tf
import numpy as np

class DQN:
    _id = -1
    def __init__(self, action_space, observation_space):
        DQN._id+=1
        self._id = DQN._id
        with tf.variable_scope('DQN_%d'%(self._id,)):
            none = [None,]
            none.extend(observation_space.shape)
            none.append(1)
            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=none)
            self._conv1 = tf.layers.conv2d(inputs=self.input, kernel_size=[1,1],
                                            filters=64, activation=tf.nn.relu)
            self._conv2 = tf.layers.conv2d(inputs=self._conv1, kernel_size=[1,1],
                                            filters=32, activation=tf.nn.relu)
            self._conv3 = tf.layers.conv2d(inputs=self._conv2, kernel_size=[1,5],
                                            filters=16, activation=tf.nn.relu)
            self._conv4 = tf.layers.conv2d(inputs=self._conv3, kernel_size=[7,1],
                                            filters=16, activation=tf.nn.relu)
            self._deep1 = tf.layers.dense(inputs=self._conv4, units=512, activation=tf.nn.relu)
            self.Q = tf.layers.dense(inputs=self._conv4, units=action_space.n, activation=None)

            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
            self._trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self._trainer.minimize(self.loss)

class experience_buffer:
    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.iter = iter(self.buffer)
        self.i=0

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
