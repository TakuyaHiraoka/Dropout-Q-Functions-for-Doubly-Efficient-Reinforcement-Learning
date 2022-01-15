import gym
import numpy as np
import torch
import time
import sys
from redq.algos.redq_sac import REDQSACAgent
from redq.algos.core import mbpo_epoches, test_agent
from redq.utils.run_utils import setup_logger_kwargs
from redq.utils.bias_utils import log_bias_evaluation
from redq.utils.logx import EpochLogger


# added by TH 20211206
import customenvs
customenvs.register_mbpo_environments()


def redq_sac(env_name, seed=0, epochs='mbpo', steps_per_epoch=1000,
             max_ep_len=1000, n_evals_per_epoch=1,
             logger_kwargs=dict(), debug=False,
             # following are agent related hyperparameters
             hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
             lr=3e-4, gamma=0.99, polyak=0.995,
             alpha=0.2, auto_alpha=True, target_entropy='mbpo',
             start_steps=5000, delay_update_steps='auto',
             utd_ratio=20, num_Q=10, num_min=2, q_target_mode='min',
             policy_update_delay=20,
             # following are bias evaluation related
             evaluate_bias=True, n_mc_eval=1000, n_mc_cutoff=350, reseed_each_epoch=True,
             # TH 20211108
             gpu_id = 0, target_drop_rate = 0.0, layer_norm = False,
             method = "redq"
             ):
    """
    :param env_name: name of the gym environment
    :param seed: random seed
    :param epochs: number of epochs to run
    :param steps_per_epoch: number of timestep (datapoints) for each epoch
    :param max_ep_len: max timestep until an episode terminates
    :param n_evals_per_epoch: number of evaluation runs for each epoch
    :param logger_kwargs: arguments for logger
    :param debug: whether to run in debug mode
    :param hidden_sizes: hidden layer sizes
    :param replay_size: replay buffer size
    :param batch_size: mini-batch size
    :param lr: learning rate for all networks
    :param gamma: discount factor
    :param polyak: hyperparameter for polyak averaged target networks
    :param alpha: SAC entropy hyperparameter
    :param auto_alpha: whether to use adaptive SAC
    :param target_entropy: used for adaptive SAC
    :param start_steps: the number of random data collected in the beginning of training
    :param delay_update_steps: after how many data collected should we start updates
    :param utd_ratio: the update-to-data ratio
    :param num_Q: number of Q networks in the Q ensemble
    :param num_min: number of sampled Q values to take minimal from
    :param q_target_mode: 'min' for minimal, 'ave' for average, 'rem' for random ensemble mixture
    :param policy_update_delay: how many updates until we update policy network
    """
    if debug: # use --debug for very quick debugging
        hidden_sizes = [2,2]
        batch_size = 2
        utd_ratio = 2
        num_Q = 3
        max_ep_len = 100
        start_steps = 100
        steps_per_epoch = 100

    assert method in ["sac", "redq", "duvn", "monosac"], "illigal method:" + str(method)
    if method == "sac":
        print("[MAIN]: use SAC. set num_Q to 2")
        num_Q = 2
    elif method == "duvn":
        print("[MAIN]: use DUVN. set num_Q  and  num_min to 1")
        num_Q = 1
        num_min = 1


    # use gpu if available
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set number of epoch
    if epochs == 'mbpo' or epochs < 0:
        # add 20211206
        mbpo_epoches['AntTruncatedObs-v2'] = 300
        mbpo_epoches['HumanoidTruncatedObs-v2'] = 300
        epochs = mbpo_epoches[env_name]
    total_steps = steps_per_epoch * epochs + 1

    """set up logger"""
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    """set up environment and seeding"""
    env_fn = lambda: gym.make(args.env)
    env, test_env, bias_eval_env = env_fn(), env_fn(), env_fn()
    # seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    # seed environment along with env action space so that everything is properly seeded for reproducibility
    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        test_env.seed(test_env_seed)
        test_env.action_space.np_random.seed(test_env_seed)
        bias_eval_env.seed(bias_eval_env_seed)
        bias_eval_env.action_space.np_random.seed(bias_eval_env_seed)
    seed_all(epoch=0)

    """prepare to init agent"""
    # get obs and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # if environment has a smaller max episode length, then use the environment's max episode length
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # we need .item() to convert it from numpy float to python float
    act_limit = env.action_space.high[0].item()
    # keep track of run time
    start_time = time.time()
    # flush logger (optional)
    sys.stdout.flush()
    #################################################################################################

    """init agent and start training"""
    agent = REDQSACAgent(env_name, obs_dim, act_dim, act_limit, device,
                 hidden_sizes, replay_size, batch_size,
                 lr, gamma, polyak,
                 alpha, auto_alpha, target_entropy,
                 start_steps, delay_update_steps,
                 utd_ratio, num_Q, num_min, q_target_mode,
                 policy_update_delay,
                 target_drop_rate=target_drop_rate, layer_norm=layer_norm) # added by TH 20211206 <- bug fix 20211207

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    for t in range(total_steps):
        # get action from agent
        a = agent.get_exploration_action(o, env)
        # Step the env, get next observation, reward and done signal
        o2, r, d, _ = env.step(a)

        # Very important: before we let agent store this transition,
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        ep_len += 1
        d = False if ep_len == max_ep_len else d

        # give new data to agent
        agent.store_data(o, a, r, o2, d)
        # let agent update
        agent.train(logger)
        # set obs to next obs
        o = o2
        ep_ret += r


        if d or (ep_len == max_ep_len):
            # store episode return and length to logger
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            # reset environment
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_agent(agent, test_env, max_ep_len, logger) # add logging here
            if evaluate_bias:
                log_bias_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)

            # reseed should improve reproducibility (should make results the same whether bias evaluation is on or not)
            if reseed_each_epoch:
                seed_all(epoch)

            """logging"""
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)

            if evaluate_bias:
                logger.log_tabular("MCDisRet", with_min_and_max=True)
                logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                logger.log_tabular("QPred", with_min_and_max=True)
                logger.log_tabular("QBias", with_min_and_max=True)
                logger.log_tabular("QBiasAbs", with_min_and_max=True)
                logger.log_tabular("NormQBias", with_min_and_max=True)
                logger.log_tabular("QBiasSqr", with_min_and_max=True)
                logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
            logger.dump_tabular()

            # flush logged information to disk
            sys.stdout.flush()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, default='Hopper-v2')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-epochs', type=int, default=-1) # -1 means use mbpo epochs
    parser.add_argument('-exp_name', type=str, default='redq_sac')
    parser.add_argument('-data_dir', type=str, default='./data/')
    parser.add_argument('-debug', action='store_true')
    # added by TH 20211108
    # use: -info, -gpu_id, -method, -target_drop_rate, -layer_norm
    parser.add_argument("-info", type=str, help="Information or name of the run")
    parser.add_argument("-gpu_id", type=int, default=0, help="GPU device ID to be used in GPU experiment, default is 1e6")
    parser.add_argument("-method", default="sac", choices=["sac", "redq", "duvn", "monosac"], help="method, default=sac")
    parser.add_argument("-target_drop_rate", type=float, default=0.0, help="drop out rate of target value function, default=0")
    parser.add_argument("-layer_norm", type=int, default=0, choices=[0, 1],
                        help="Using layer normalization for training critics if set to 1 (TH), default=0")
    # ignore: -eval_every, -frames, -eval_runs, -updates_per_step, -target_entropy,
    parser.add_argument("-eval_every", type=int, default=1000,
                        help="Number of interactions after which the evaluation runs are performed, default = 1000")
    parser.add_argument("-frames", type=int, default= 1000000,
                        help="The amount of training interactions with the environment, default is 1mio")
    parser.add_argument("-eval_runs", type=int, default=3, help="Number of evaluation runs performed, default = 1")
    parser.add_argument("-updates_per_step", type=int, default=1, help="Number of training updates per one environment step, default = 1")
    parser.add_argument("-target_entropy", type=float, default=None, help="target entropy , default=Num action")

    args = parser.parse_args()

    # modify the code here if you want to use a different naming scheme
    exp_name_full = args.exp_name + '_%s' % args.env

    # override log directory path. TH 20211108
    args.data_dir  = './runs/' + str(args.info) + '/'

    # specify experiment name, seed and data_dir.
    # for example, for seed 0, the progress.txt will be saved under data_dir/exp_name/exp_name_s0
    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)


    redq_sac(args.env, seed=args.seed, epochs=args.epochs,
             logger_kwargs=logger_kwargs, debug=args.debug,
             # added by TH 20211206
             gpu_id=args.gpu_id,
             target_drop_rate=args.target_drop_rate, # tagert entropy -> dropout rate. Fixed 20211206
             layer_norm=bool(args.layer_norm),
             method=args.method)
    #redq_sac(args.env, seed=args.seed, epochs=args.epochs,
    #         logger_kwargs=logger_kwargs, debug=args.debug)
