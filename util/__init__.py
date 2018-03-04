import numpy as np
import argparse

"""
    Finds tensorflow variable named in a same way as one supplied.
"""
def map_dqn_var(var, to_iterable_vars, var_scope = "DQN_#/", iterable_scope = "DQN_#/"):
    return next(filter(lambda x: x.name[len(iterable_scope+'/'):] == var.name[len(var_scope+'/'):], to_iterable_vars))

"""
    Buffer with limited size.
    If added items > size then overwrites the oldest items.
"""
class experience_buffer:
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.iter = iter(self.buffer)
        self.i=0

    def add(self, experience):
        if np.asarray(experience).ndim == 1:
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
                return np.asarray(np.random.permutation(self.buffer)[0:size])
            else:
                return np.asarray(np.random.permutation(self.buffer)[0:size])
        except ValueError:
            print(size <= len(self.buffer))
            raise

    def full(self):
        return len(self.buffer) == self.buffer_size

    def erase(self):
        self.buffer = []


"""
    Given the argparse namespace, creates a string with text representations of
    parameters joined with delimeter.
"""
def build_param_string(param_namespace, delimiter=","):
    param_string = []
    param_string.append("env_name={}".format(param_namespace.env_name))
    for key,val in vars(param_namespace).items():
        if type(val) is not str:
            param_string.append("{}={}".format(key, val))
    return delimiter.join(param_string)

"""
    Given the argparse namespace with lists inside, unrolls it
    and returns list of shallow parameter namespaces.
"""
def create_list_of_param_namespaces(param_namespace_with_lists):
    list_of_params = []
    #unrolls everything
    for key, value in vars(param_namespace_with_lists).items():
        if hasattr(value, '__iter__') and type(value) is not str:
            ps_to_add = []
            for x in value:
                if not list_of_params:
                    p = argparse.Namespace(**vars(param_namespace_with_lists))
                    setattr(p, key, x)
                    list_of_params.append(p)
                else:
                    for ps in list_of_params:
                        p = argparse.Namespace(**vars(ps))
                        setattr(p, key, x)
                        ps_to_add.append(p)
            list_of_params.extend(ps_to_add)
    final_list = []
    # picks final shallow namespaces
    for ps in list_of_params:
        good = True
        for key, value in vars(ps).items():
            if hasattr(value, '__iter__') and type(value) is not str:
                good = False
        if good:
            final_list.append(ps)
    return final_list
