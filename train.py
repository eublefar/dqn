from gym import spaces
import gym
import tensorflow as tf
import numpy as np
import dqn
from util import experience_buffer
from util.image_preproc import preprocess

def train(env_name = 'SpaceInvaders-v0',
        render = False,
        save_dir = 'checkpoints/',
        summary_dir = '/tmp/dqn/',
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
    env.mode = "batch"
    eps = eps_high
    eps_step = (eps_high - eps_low)/eps_degrade_steps
    # Reproducibility for the win
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    observation_shape = list(env.observation_space.shape)
# we replace image color channels with 4 sequential grayscale images
    observation_shape[-1] = 4
# shape after downsampling
    observation_shape[0] /= 2
    observation_shape[1] /= 2
    target = dqn.DQN(observation_shape, env.action_space.n,
                    batch_size=batch_size, pixels = True, trainable = False)
    main = dqn.DQN(observation_shape, env.action_space.n,
                    batch_size=batch_size, pixels = True)

    exp_buffer = experience_buffer()

    apply_update_op = target.applyUpdate(main, tau)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    writer = tf.summary.FileWriter(summary_dir)
    writer.add_graph(tf.get_default_graph())
    for var in tf.trainable_variables(scope="DQN_.*"):
        tf.summary.histogram(var.name, var)
    total_reward = tf.Variable(initial_value=0, trainable=False,
                               name="total_reward", dtype=tf.float32)
    tf.summary.scalar(total_reward.name, total_reward)
    summary_op = tf.summary.merge_all()

    num_steps = 0


    with tf.Session() as session:
        session.run(init)

        if restore_dir is not None:
            saver.restore(session, restore_dir)

        last_observations = experience_buffer(buffer_size=4)

        for episode in range(num_episodes):
            # Prepaire for the episode
            last_observations = experience_buffer(buffer_size=4)
            last_observations_next = experience_buffer(buffer_size=4)
            episode_buffer = experience_buffer()
            last_observations.add((preprocess(env.reset()),))
            ep_steps = 0
            done = False
            session.run(total_reward.assign(0))

            if episode%checkpoint_episode_num == 0:
                saver.save(session, save_dir)

            while not done:
                num_steps+=1
                ep_steps+=1
                if num_steps <= num_pretrain_steps \
                   or np.random.random_sample() < eps \
                   or not last_observations.full():
                    action = env.action_space.sample()
                else:
                    action = np.argmax(session.run(main.Q_values,
                    feed_dict={main.input:
                                np.expand_dims(
                                np.stack(last_observations.buffer, axis=2),
                                axis =0)}))

                obs_next, reward, done, _ = env.step(action)
                obs_next = preprocess(obs_next)
                last_observations_next.add((obs_next,))

                if last_observations.full() and last_observations_next.full():
                    episode_buffer.add( (np.stack(last_observations.buffer, axis=2),
                                        action,
                                        reward,
                                        np.stack(last_observations_next.buffer, axis=2),
                                        done))
                if num_steps >= num_pretrain_steps:
                    batch = exp_buffer.sample(batch_size)
                    next_Q = session.run(target.Q_values,
                    feed_dict={target.input:np.stack(batch[:,3])})
                    targetQ = np.where(batch[:,4], batch[:,2], batch[:,2] + discount*np.max(next_Q, 1))
                    session.run(main.updateModel,
                    feed_dict={
                        main.input:np.stack(batch[:,0]),
                        main.targetQ: targetQ,
                        main.actions:batch[:,1]
                    })

                    session.run(apply_update_op)
                    eps -= eps_step

                last_observations.add((obs_next,))
                if render:
                    env.render()
                if ep_steps >= num_steps_per_episode:
                    done = True
                session.run(tf.assign_add(total_reward, reward))
            exp_buffer.add(episode_buffer.buffer)
            if num_steps >= num_pretrain_steps:
                summary = session.run(summary_op)
                writer.add_summary(summary, num_steps)
