
def map_dqn_var(var, to_iterable_vars):
    return next(filter(lambda x: x.name[len('DQN_#/'):] == var.name[len('DQN_#/'):], to_iterable_vars))
