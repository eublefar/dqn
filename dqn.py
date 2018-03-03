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
            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=batch_shape,
                                        name="input")
            outp = self._defineModel(self.input)
            self.Q_values = tf.layers.dense(inputs=outp,
                                            units=action_space.n,
                                            activation=None,
                                            name='output')
            self.targetQ = tf.placeholder(shape=[None],
                                          dtype=tf.float32,
                                          name="targetQ_ph")
            self.actions = tf.placeholder(dtype=tf.float32,
                                          shape=[None],
                                          name='actions')
            self.actions_onehot = tf.one_hot(tf.to_int32(self.actions),
                                             action_space.n,
                                             axis=1,
                                             dtype=tf.float32)
            self.Q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions_onehot),
                                               axis=1)
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
        conv1 = tf.layers.conv2d(inputs=inp, kernel_size=[7,7],
                                filters=32, activation=tf.nn.relu, name='conv1')
        max_pool1 = tf.layers.max_pooling2d(conv1, [3,3], [3,3], 'same')
        conv2 = tf.layers.conv2d(inputs=max_pool1, kernel_size=[5,5],
                                filters=32, activation=tf.nn.relu, name='conv2')
        max_pool2 = tf.layers.max_pooling2d(conv2, [2,2], [2,2], 'same')
        conv4 = tf.layers.conv2d(inputs=max_pool2, kernel_size=[3,3],
                                filters=64, activation=tf.nn.relu, name='conv4')
        max_pool3 = tf.layers.max_pooling2d(conv4, [2,2], [2,2], 'same')
        flat = tf.layers.flatten(max_pool3)
        dense1 = tf.layers.dense(inputs=flat, units=2048,
                                activation=tf.nn.relu, name='dense1')
        return dense1

    def _defineModel_noConv(self, inp):
        dense1 = tf.layers.dense(inputs=inp, units=512,
                                activation=tf.nn.relu, name='dense1')
        dense2 = tf.layers.dense(inputs=dense1, units=512,
                                activation=tf.nn.relu, name='dense2')
        return dense2
