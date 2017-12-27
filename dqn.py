import tensorflow as tf
import numpy as np

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
            batch_shape = [batch_size,] + list(observation_space.shape)
            self.input = tf.placeholder(dtype=tf.float32, shape=batch_shape)
            outp = self._defineModel(self.input)
            self.Q = tf.layers.dense(inputs=outp, units=action_space.n,
                                        activation=None, name='output')
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
            self._trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self._trainer.minimize(self.loss)

    def applyUpdate(self, source, tau):
        src_vars = tf.trainable_variables(scope="DQN_%d"%(source.id))
        target_vars = tf.trainable_variables(scope="DQN_%d"%(self.id))
        for i in range(len(src_vars)):
            var = target_vars.find(
            lambda x:
            x.name[len('DQN_#/'):] == src_vars[i].name[len('DQN_#/'):]
            )
            var.assign(var.value() * (1 - tau) + src_vars[i] * tau)

    def _defineModel(self, inp):
        inp = tf.reshape(tensor=inp, shape=inp.get_shape().as_list()+[1,])
        conv1 = tf.layers.conv2d(inputs=inp, kernel_size=[1,1],
                                filters=64, activation=tf.nn.relu, name='conv1')
        conv2 = tf.layers.conv2d(inputs=conv1, kernel_size=[1,1],
                                filters=32, activation=tf.nn.relu, name='conv2')
        conv3 = tf.layers.conv2d(inputs=conv2, kernel_size=[1,5],
                                filters=16, activation=tf.nn.relu, name='conv3')
        conv4 = tf.layers.conv2d(inputs=conv3, kernel_size=[7,1],
                                filters=16, activation=tf.nn.relu, name='conv4')
        dense1 = tf.layers.dense(inputs=conv4, units=512,
                                activation=tf.nn.relu, name='dense1')
        return _dense1

    def _defineModel_noConv(self, inp):
        dense1 = tf.layers.dense(inputs=inp, units=512,
                                activation=tf.nn.relu, name='dense1')
        dense2 = tf.layers.dense(inputs=dense1, units=512,
                                activation=tf.nn.relu, name='dense2')
        return dense2

class experience_buffer:
    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.iter = iter(self.buffer)
        self.i=0

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
