import argparse
from train import train

parser = argparse.ArgumentParser(description="""Deep Q network executed
                                            on Atari gym environments""")
parser.add_argument("--restore-dir", nargs=1, dest="restore_dir", type=str,
                default=None,
                help="""If set, tries to restore model
                        from the checkpoints in directory specified""")
parser.add_argument("--checkpoint-dir", nargs=1, dest="checkpoint_dir", type=str
                default="/tmp/dqn/1",
                help="""Specifies the directory to which checkpoints are saved.
                        By default checkpoints are saved in /tmp/dqn/1 """)
parser.add_argument("--episodes-per-checkpoints", nargs="+", dest="checkpoint_episode_num", type=int,
                default=4,
                help="""Checkpoints are created per this number of episodes""")
parser.add_argument("--render", nargs=1, dest="restore_dir", type=bool,
                default=False,
                help="If true, renders the environment")
parser.add_argument("--num-episodes", nargs=1, dest="num_episodes", type=int,
                default=10000,
                help="""number of episodes to run""")
parser.add_argument("--num-pretrain-steps", nargs=1, dest="num_pretrain_steps", type=int,
                default=10000,
                help="""number of episodes to run with random policy
                        before training model(to populate experience buffer)""")
parser.add_argument("--epsilon-max", nargs="+", dest="eps_max", type=float,
                default=1.,
                help="""Maximum value of epsilon parameter for degrading epsilon-greedy policy""")
parser.add_argument("--epsilon-min", nargs="+", dest="eps_min", type=float,
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
