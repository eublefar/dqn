from gym import spaces
import gym
import tensorflow as tf
import dqn

env = gym.make('MountainCar-v0')
batch_size=5
tau = 0.001
env.observation_space
target = dqn.DQN(env.observation_space, env.action_space, batch_size=batch_size)
main = dqn.DQN(env.observation_space, env.action_space, batch_size=batch_size)
exp_buffer = dqn.experience_buffer()
obs = env.reset()
for i in range(batch_size):
    exp_buffer.add(env.step(env.action_space.sample()))

exp_buffer.buffer

for x in xrange(10000):
