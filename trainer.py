from gym import spaces
import gym
import tensorflow as tf
import numpy as np
import dqn
from util import experience_buffer
from util.image_preproc import preprocess
import _pickle as pcl
import os.path


class Trainer:

    @classmethod
    def create_from_namespace(cls, parsed_args, session):
        return cls(
                    env_name = parsed_args.env_name,
                    render = parsed_args.render,
                    save_dir = parsed_args.save_dir,
                    summary_dir = parsed_args.summary_dir,
                    eps_high = parsed_args.eps_high,
                    eps_low = parsed_args.eps_low,
                    eps_degrade_steps = parsed_args.eps_degrade_steps,
                    batch_size = parsed_args.batch_size,
                    tau = parsed_args.tau,
                    discount = parsed_args.discount,
                    seed = parsed_args.seed,
                    session = session)


    def __init__(self,
                env_name,
                render,
                save_dir,
                summary_dir,
                eps_high,
                eps_low,
                eps_degrade_steps,
                batch_size,
                tau,
                discount,
                seed,
                session,
                pickle_file = "buffers.pkl"):
        self.env = gym.make(env_name)
        self.env.mode = "batch"
        self.seed(seed)
        self.eps_low = eps_low
        self.eps,self.eps_step = \
            self.get_epsilon_update(eps_high,
                                    eps_low,
                                    eps_degrade_steps)

        observation_shape = self.get_observation_shape()
        self.target = dqn.DQN(observation_shape,
                              self.env.action_space.n,
                              batch_size=batch_size,
                              pixels = True,
                              trainable = False)
        self.main = dqn.DQN(observation_shape,
                            self.env.action_space.n,
                            batch_size=batch_size,
                            pixels = True)
        self.exp_buffer = experience_buffer(buffer_size=1000000)
        self.apply_update_op = self.target.applyUpdate(self.main, tau)
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(summary_dir)
        self.writer.add_graph(tf.get_default_graph())
        self.summary_op = self.get_summary_op()
        self.session = session
        self.tf_vars_initiated = False
        self.save_dir = save_dir
        self.history_len = 4
        self.batch_size = batch_size
        self.discount = discount
        self.pickle_file = pickle_file

    def seed(self, seed):
        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

    def get_epsilon_update(self, eps_high, eps_low, steps):
        eps = eps_high
        eps_step = (eps_high - eps_low)/steps
        return eps, eps_step

    def get_observation_shape(self):
        observation_shape = list(self.env.observation_space.shape)
        # we replace image color channels with 4 sequential grayscale images
        observation_shape[-1] = 4
        # shape after downsampling
        observation_shape[0] /= 2
        observation_shape[1] /= 2
        return observation_shape

    def get_summary_op(self):
        for var in tf.trainable_variables(scope="DQN_.*"):
            tf.summary.histogram(var.name, var)

        self.total_reward = tf.Variable(initial_value=0, trainable=False,
                                   name="total_reward", dtype=tf.float32)
        tf.summary.scalar(self.total_reward.name, self.total_reward)
        return tf.summary.merge_all()

    def init_variables(self):
        self.session.run(self.init_op)
        self.tf_vars_initiated = True

    def restore(self, restore_dir):
        self.saver.restore(self.session, restore_dir)
        self.tf_vars_initiated = True
        exp_buffer_pickle = restore_dir + '/' + self.pickle_file
        if os.path.isfile(exp_buffer_pickle):
            with open(exp_buffer_pickle, 'r') as f:
                self.exp_buffer, self.eps, self.eps_step = pcl.load(f)

    def train(self,
              num_episodes,
              num_pretrain_steps = 10000,
              num_steps_per_episode = 10000,
              checkpoint_episode_num = 10):
        try:
            self._train(num_episodes,
                        num_pretrain_steps,
                        num_steps_per_episode,
                        checkpoint_episode_num)
        except KeyboardInterrupt:
            self.handle_interrupt()

    def handle_interrupt(self):
        with open(self.save_dir + '/' + self.pickle_file, 'w+') as p_file:
            pcl.dump((self.exp_buffer, self.eps, self.eps_step), p_file)
        self.saver.save(self.session, self.save_dir)

    def _train(self,
              num_episodes,
              num_pretrain_steps = 10000,
              num_steps_per_episode = 10000,
              checkpoint_episode_num = 10):
        if not self.tf_vars_initiated:
            raise AssertionError("""
                        Tensorflow variables were not initiated!
                        Use init_variables or restore methods before training
                                """)
        self.step = 0
        self.last_observations = experience_buffer(buffer_size=self.history_len)
        self.last_observations_next = experience_buffer(buffer_size=self.history_len)
        self.episode_buffer = experience_buffer(buffer_size=num_steps_per_episode)
        self.num_pretrain_steps = num_pretrain_steps
        for episode in range(num_episodes):
            self.episode_step = 0
            self.episode_buffer.erase()

            first_obs = preprocess(self.env.reset())
            done = False
            self.last_observations.buffer = list((np.zeros(first_obs.shape),)\
                                                  * self.history_len)
            self.last_observations_next.buffer = list((np.zeros(first_obs.shape),)\
                                                  * self.history_len)
            self.last_observations.add((first_obs,))
            self.last_observations_next.add((first_obs,))

            if episode % checkpoint_episode_num == 0 and episode != 0:
                self.saver.save(self.session, self.save_dir)

            while not done:
                self.step+=1
                self.episode_step+=1
                action = self.generate_action()
                obs_next, reward, done, _ = self.env.step(action)
                if self.episode_step >= num_steps_per_episode:
                    done = True
                obs_next = preprocess(obs_next)
                self.last_observations_next.add((obs_next,))
                self.episode_buffer.add(
                                    (np.stack(self.last_observations.buffer,
                                              axis=2),
                                    action,
                                    reward,
                                    np.stack(self.last_observations_next.buffer,
                                             axis=2),
                                    done))
                if self.step >= num_pretrain_steps:
                    self.update_weights()
                self.last_observations.add((obs_next,))
            self.exp_buffer.add(self.episode_buffer.buffer)
            if self.step >= num_pretrain_steps:
                self.write_summaries()


    def generate_action(self):
        action = None
        if self.step <= self.num_pretrain_steps \
           or np.random.random_sample() < self.eps:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.session.run(self.main.Q_values,
            feed_dict={self.main.input:
                        np.expand_dims(
                        np.stack(self.last_observations.buffer, axis=2).astype(np.float32),
                        axis =0)}))
        return action

    def update_weights(self):
        batch = self.exp_buffer.sample(self.batch_size)
        next_Q = self.session.run(self.target.Q_values,
                             feed_dict={
                                 self.target.input : np.stack(batch[:,3]).astype(np.float32)
                             })
        targetQ = np.where(batch[:,4], batch[:,2], batch[:,2] + self.discount*np.max(next_Q, 1))
        self.session.run(self.main.updateModel,
                    feed_dict={
                        self.main.input:   np.stack(batch[:,0]).astype(np.float32),
                        self.main.targetQ: targetQ,
                        self.main.actions: batch[:,1]
                    })
        self.session.run(self.apply_update_op)
        if self.eps > self.eps_low:
            self.eps -= self.eps_step

    def write_summaries(self):
        self.update_summary_vars()
        summary = self.session.run(self.summary_op)
        self.writer.add_summary(summary, self.step)

    def update_summary_vars(self):
        self.session.run(tf.assign(self.total_reward,
                        np.sum(np.asarray(self.episode_buffer.buffer)[:,2])))
