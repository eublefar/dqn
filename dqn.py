import tensorflow as tf
import numpy as np
import util

class DQN:

    _id = -1

    def __init__(self, observation_space_shape, action_number,
                batch_size=15, learning_rate = 0.0001, pixels = False, trainable = True):
        DQN._id+=1
        self.id = DQN._id

        if not pixels:
            self._defineModel = self._defineModel_noConv

        with tf.variable_scope('DQN_%d'%(self.id,)):
            self.batch_size = batch_size
            batch_shape = [None,] + list(observation_space_shape)
            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=batch_shape,
                                        name="input")
            outp = self._defineModel(self.input, trainable = trainable)
            self.Q_values = tf.layers.dense(inputs=outp,
                                            units=action_number,
                                            activation=None,
                                            name='output',
                                            trainable = trainable)
            self.targetQ = tf.placeholder(shape=[None],
                                          dtype=tf.float32,
                                          name="targetQ_ph")
            self.actions = tf.placeholder(dtype=tf.float32,
                                          shape=[None],
                                          name='actions')
            self.actions_onehot = tf.one_hot(tf.to_int32(self.actions),
                                             action_number,
                                             axis=1,
                                             dtype=tf.float32)
            self.Q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions_onehot),
                                               axis=1)
            if trainable:
                self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
                self._trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.updateModel = self._trainer.minimize(self.loss)

    def applyUpdate(self, source, tau):
        src_vars = tf.trainable_variables(scope="DQN_%d"%(source.id))
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DQN_%d"%(self.id))
        update_ops = []
        for i in range(len(src_vars)):
            var = util.map_dqn_var(src_vars[i], target_vars)
            op = var.assign(var.value() * (1 - tau) + src_vars[i] * tau)
            update_ops.append(op)
        return update_ops

    def _defineModel(self, inp, trainable = True):
        # inp = tf.reshape(tensor=inp, shape=inp.get_shape().as_list()+[1,])
        conv1 = tf.layers.conv2d(inputs=inp, kernel_size=[8,8], strides=(3,3),
                                filters=16, activation=tf.nn.relu, name='conv1',
                                trainable = trainable)
        conv2 = tf.layers.conv2d(inputs=conv1, kernel_size=[4,4],
                                filters=32, activation=tf.nn.relu, name='conv2',
                                trainable = trainable)
        flat = tf.layers.flatten(conv2)
        dense1 = tf.layers.dense(inputs=flat, units=256,
                                activation=tf.nn.relu, name='dense1',
                                trainable = trainable)
        return dense1

    def _defineModel_noConv(self, inp):
        dense1 = tf.layers.dense(inputs=inp, units=512,
                                activation=tf.nn.relu, name='dense1')
        dense2 = tf.layers.dense(inputs=dense1, units=512,
                                activation=tf.nn.relu, name='dense2')
        return dense2
