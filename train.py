from gym import spaces
import tensorflow as tf
import dqn
none = [None,]
none.extend(spaces.Box(shape=(18,31),high=101, low=51).shape)
none
dqn.DQN(spaces.Discrete(10), spaces.Box(shape=(1,2),high=101, low=51))
