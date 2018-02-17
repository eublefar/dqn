
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
        discount = 0.8):
    """Stub train function."""
    print({
            "env_name":env_name,
            "render":render,
            "save_dir":save_dir,
            "summary_dir":summary_dir,
            "restore_dir":restore_dir,
            "num_episodes":num_episodes,
            "num_pretrain_steps":num_pretrain_steps,
            "num_steps_per_episode":num_steps_per_episode,
            "eps_high":eps_high,
            "eps_low":eps_low,
            "eps_degrade_steps":eps_degrade_steps,
            "checkpoint_episode_num":checkpoint_episode_num,
            "batch_size":batch_size,
            "tau":tau,
            "discount":discount
    })
