from gym import spaces
import gym
import tensorflow as tf
import dqn
import numpy as np


if __name__=="__main__":

    env = gym.make('SpaceInvaders-v0')

    restore_model = False
    restore_dir = 'checkpoints/checkpoint_0.ckpt'
    num_episodes = 10000
    num_pretrain_steps = 10000
    num_steps_per_episode = 100000
    eps_high = 1.
    eps_low = 0.1
    eps_degrade_steps = 10000
    eps = eps_high
    eps_step = (eps_high - eps_low)/eps_degrade_steps
    checkpoint_episode_num = 50
    batch_size=10
    tau = 0.001
    discound = 0.8

    target = dqn.DQN(env.observation_space, env.action_space,
                    batch_size=batch_size, pixels = True)
    main = dqn.DQN(env.observation_space, env.action_space,
                    batch_size=batch_size, pixels = True)
    exp_buffer = dqn.experience_buffer()

    apply_update_op = target.applyUpdate(main, tau)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('/tmp/dqn/1')

    num_steps = 0
# Reproductableness for the win
    seed = 7
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    with tf.Session() as session:
        session.run(init)
        if restore_model:
            ckpt = tf.train.get_checkpoint_state(restore_dir)
            saver.restore(session,ckpt.model_checkpoint_path)

        writer.add_graph(session.graph)
        for var in tf.global_variables():
            tf.summary.scalar(var.name, var)
        summary_op = tf.summary.merge_all()

        for episode in range(num_episodes):
            episode_buffer = dqn.experience_buffer()
            obs = env.reset()
            ep_steps = 0
            done = False
            if episode%checkpoint_episode_num == 0:
                saver.save(session, restore_dir)
            while not done:
                num_steps+=1
                ep_steps+=1

                if num_steps <= num_pretrain_steps or np.random.random_sample() < eps:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(session.run(main.Q_output, feed_dict={main.input:obs}))

                obs_next, reward, done, _ = env.step(action)
                episode_buffer.add((obs,action,reward,obs_next))

                if num_steps >= num_pretrain_steps:
                    try:
                        batch = exp_buffer.sample(batch_size)
                        next_Q = session.run(target.Q_output,
                        feed_dict={target.input:np.stack(batch[:,3])})
                        targetQ = batch[:,2] + discound*np.max(next_Q, 1)
                        session.run(apply_update_op)
                        session.run(main.updateModel,
                        feed_dict={
                            main.input:np.stack(batch[:,0]),
                            main.targetQ: targetQ,
                            main.actions:batch[:,1]
                        })
                        eps -= eps_step
                    except ValueError:
                        print_debug_info()
                        raise

                obs = obs_next

                if ep_steps >= num_steps_per_episode:
                    done = True
                # summary = session.run(summary_op)
                # writer.add_summary(summary, num_steps)
                # env.render()
            exp_buffer.add(episode_buffer)

def print_debug_info():
    print ("""
                Number of steps = {1}\n
                Episode steps = {2}\n
                experience batch shape = {3}\n
                Q_output shape = {4}\n
                batch reward sum = {5}\n
            """.format(num_steps,
                ep_steps,
                batch[:,3].shape,
                target.Q.get_shape().as_list(),
                np.sum(batch[:,2])))
