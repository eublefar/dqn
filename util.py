import numpy as np

def map_dqn_var(var, to_iterable_vars):
    return next(filter(lambda x: x.name[len('DQN_#/'):] == var.name[len('DQN_#/'):], to_iterable_vars))


class experience_buffer:
    def __init__(self, buffer_size=100000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.iter = iter(self.buffer)
        self.i=0

    def add(self, experience):
        if np.array(experience).ndim == 1:
            if len(self.buffer) + 1 >= self.buffer_size:
                self.buffer[0:1] = []
            self.buffer.append(experience)
        else:
            if len(self.buffer) + len(experience) >= self.buffer_size:
                self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
            self.buffer.extend(experience)

    def sample(self,size):
        try:
            if size <= len(self.buffer):
                return np.reshape(np.array(np.random.permutation(self.buffer)[0:size]),[size,4])
            else:
                return np.reshape(np.array(np.random.permutation(self.buffer)[0:size]),[-1,4])
        except ValueError:
            print(size <= len(self.buffer))
            raise
