from gym import spaces
import gym
import tensorflow as tf
import numpy as np
import dqn
from util import experience_buffer

def train(env_name = 'SpaceInvaders-v0',
        render = False,
        save_dir = 'checkpoints/checkpoint_0.ckpt',
        summary_dir = '/tmp/dqn/1',
        restore_dir = None,
        num_episodes = 10000,
        num_pretrain_steps = 10000,
        num_steps_per_episode = 100000,
        eps_high = 1.,
        eps_low = 0.1,
        eps_degrade_steps = 500000,
        checkpoint_episode_num = 50,
        batch_size=10,
        tau = 0.001,
        discount = 0.8,
        seed = 7):


    env = gym.make(env_name)
    eps = eps_high
    eps_step = (eps_high - eps_low)/eps_degrade_steps
    # Reproducibility for the win
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    target = dqn.DQN(env.observation_space, env.action_space,
                    batch_size=batch_size, pixels = True)
    main = dqn.DQN(env.observation_space, env.action_space,
                    batch_size=batch_size, pixels = True)

    exp_buffer = experience_buffer()
    apply_update_op = target.applyUpdate(main, tau)
    total_reward = tf.Variable(initial_value=0, trainable=False,
                               name="total_reward", dtype=tf.float32)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(summary_dir)
    writer.add_graph(tf.get_default_graph())
    for var in tf.trainable_variables(scope="DQN_.*"):
        tf.summary.histogram(var.name, var)
    tf.summary.scalar(total_reward.name, total_reward)
    summary_op = tf.summary.merge_all()

    num_steps = 0


    with tf.Session() as session:
        session.run(init)

        if restore_dir is not None:
            saver.restore(session, restore_dir)
        for episode in range(num_episodes):
            episode_buffer = dqn.experience_buffer()
            obs = env.reset()
            ep_steps = 0
            done = False
            session.run(total_reward.assign(0))
            if episode%checkpoint_episode_num == 0:
                saver.save(session, restore_dir)
            while not done:
                num_steps+=1
                ep_steps+=1
                if num_steps <= num_pretrain_steps or np.random.random_sample() < eps:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(session.run(main.Q_values,
                    feed_dict={main.input:obs.reshape((1,) + obs.shape)}))
                obs_next, reward, done, _ = env.step(action)
                session.run(tf.assign_add(total_reward, reward))
                episode_buffer.add((obs,action,reward,obs_next))

                if num_steps >= num_pretrain_steps:
                    try:
                        batch = exp_buffer.sample(batch_size)
                        next_Q = session.run(target.Q_values,
                        feed_dict={target.input:np.stack(batch[:,3])})
                        targetQ = batch[:,2] + discount*np.max(next_Q, 1)
                        session.run(main.updateModel,
                        feed_dict={
                            main.input:np.stack(batch[:,0]),
                            main.targetQ: targetQ,
                            main.actions:batch[:,1]
                        })

                        session.run(apply_update_op)
                        eps -= eps_step
                    except ValueError:
                        print ("""
                                    Number of steps = {1}\n
                                    Episode steps = {2}\n
                                    experience batch shape = {3}\n
                                    Q_output shape = {4}\n
                                    batch reward sum = {5}\n
                                """.format(num_steps,
                                    ep_steps,
                                    batch[:,3].shape,
                                    np.sum(batch[:,2])))
                        raise
                obs = obs_next
                if render:
                    env.render()
                if ep_steps >= num_steps_per_episode:
                    done = True
            exp_buffer.add(episode_buffer.buffer)
            if num_steps >= num_pretrain_steps:
                summary = session.run(summary_op)
                writer.add_summary(summary, num_steps)
