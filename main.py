import argparse
# from train import train
from util.stubs import train

parser = argparse.ArgumentParser(description="""Deep Q network executed
                                            on Atari gym environments""")
parser.add_argument("--render", nargs=1, dest="render", type=bool,
                default=False,
                help="If true, renders the environment")
parser.add_argument("--num-episodes", nargs=1, dest="num_episodes", type=int,
                default=10000,
                help="""number of episodes to run""")
parser.add_argument("--num-steps-per-episode", nargs=1, dest="num_steps_per_episode", type=int,
                default=10000,
                help="""number of episodes to run""")
parser.add_argument("--num-pretrain-steps", nargs=1, dest="num_pretrain_steps", type=int,
                default=10000,
                help="""number of episodes to run with random policy
                        before training model(to populate experience buffer)""")
parser.add_argument("--save-dir", nargs=1, dest="save_dir", type=str,
                default="checkpoints/",
                help="""Specifies the directory to which checkpoints are saved.
                        By default checkpoints are saved in checkpoints/.
                        Directory si augmented with param string if such exists""")
parser.add_argument("--summary-dir", nargs=1, dest="summary_dir", type=str,
                default="/tmp/dqn/1",
                help="""Specifies the main directory where summaries are saved.
                        Directory is augmented with param string if such exists""")
parser.add_argument("--restore-dir", nargs="+", dest="restore_dir", type=str,
                default=None,
                help="""If set, tries to restore model
                        from the checkpoints in directory specified.
                        If array supplied script will restore dirs on the go""")
parser.add_argument("--env-name", nargs='+', dest="env_name", type=str,
                default='SpaceInvaders-v0',
                help="""Name of the environment to solve by DQN""")
parser.add_argument("--episodes-per-checkpoints", nargs="+", dest="checkpoint_episode_num", type=int,
                default=4,
                help="""Checkpoints are created per this number of episodes""")
parser.add_argument("--epsilon-max", nargs="+", dest="eps_high", type=float,
                default=1.,
                help="""Maximum value of epsilon parameter for degrading epsilon-greedy policy""")
parser.add_argument("--epsilon-min", nargs="+", dest="eps_low", type=float,
                default=0.1,
                help="""Minimum value of epsilon parameter for degrading epsilon-greedy policy""")
parser.add_argument("--eps-degrade-steps", nargs="+", dest="eps_degrade_steps", type=int,
                default=100000,
                help="""Steps for epsilon parameter to degrade""")
parser.add_argument("--batch-size", nargs="+", dest="batch_size", type=int,
                default=10,
                help="""Batch size of experiences to train on""")
parser.add_argument("--update-coef", nargs="+", dest="tau", type=float,
                default=0.001,
                help="""Target network update rate""")
parser.add_argument("--discount", nargs="+", dest="discount", type=float,
                default=0.9,
                help="""Reward time discount. Ensures finite total reward.""")
params = parser.parse_args()



if __name__=="__main__":
    list_of_params = []
    for key, value in vars(params).items():
        if hasattr(value, '__iter__') and type(value) is not str:
            for x in value:
                p = argparse.Namespace(**vars(params))
                setattr(p, key, x)
                list_of_params.append(p)

    for v in list_of_params:
        print("Starting training with parameters: {}".format(v))
        train(v.env_name,
            v.render,
            v.save_dir,
            v.summary_dir,
            v.restore_dir,
            v.num_episodes,
            v.num_pretrain_steps,
            v.num_steps_per_episode,
            v.eps_high,
            v.eps_low,
            v.eps_degrade_steps,
            v.checkpoint_episode_num,
            v.batch_size,
            v.tau,
            v.discount)
