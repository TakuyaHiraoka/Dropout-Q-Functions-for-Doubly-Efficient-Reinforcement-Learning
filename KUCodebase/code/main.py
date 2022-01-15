import os
import argparse
import datetime
import gym

from agent import SacAgent

# TODO use shared util.utilTH in SAC-extention
from util.utilsTH import SparseRewardEnv

# TODO remove IQN agent part
#from IQNagent import IQNSacAgent

import customenvs
customenvs.register_mbpo_environments()

from agent4profile import SacAgent4Profile


def run():
    parser = argparse.ArgumentParser()
    # replaced wtih SAC-extention args 20210705
    parser.add_argument("-env", type=str, default="HalfCheetah-v2",
                        help="Environment name, default = HalfCheetahBulletEnv-v0")
    parser.add_argument('-seed', type=int, default=0)
    #added byTH 20210705
    # common
    parser.add_argument("-info", type=str, help="Information or name of the run")
    parser.add_argument("-frames", type=int, default=1_000_000,
                        help="The amount of training interactions with the environment, default is 1mio")
    parser.add_argument("-gpu_id", type=int, default=0,
                        help="GPU device ID to be used in GPU experiment, default is 1e6")
    # evaluation
    parser.add_argument("-eval_every", type=int, default=1000,
                        help="Number of interactions after which the evaluation runs are performed, default = 1000")
    parser.add_argument("-eval_runs", type=int, default=3, help="Number of evaluation runs performed, default = 1")
    # sparse env
    parser.add_argument("-sparsity_th", type=float, default=0.0,
                        help="threshold for make reward sparse (i.e., lambda in PolyRL paper), default is 0.0")
    # stabilization
    parser.add_argument("-huber", type=int, default=0, choices=[0, 1],
                        help="Using Huber loss for training critics if set to 1 (TH), default=0") # TODO remove
    parser.add_argument("-layer_norm", type=int, default=0, choices=[0, 1],
                        help="Using layer normalization for training critics if set to 1 (TH), default=0")
    # multi-step and per
    parser.add_argument("-n_step", type=int, default=1, help="Using n-step bootstrapping, default=1")
    parser.add_argument("-per", type=int, default=0, choices=[0, 1],
                        help="Adding Priorizied Experience Replay to the agent if set to 1, default = 0")
    # dist RL added @ 20210711
    parser.add_argument("-dist", "--distributional", type=int, default=0, choices=[0, 1],
                        help="Using a distributional IQN Critic if set to 1, default=0") # TODO remove
    # learning per steps
    parser.add_argument("-updates_per_step", type=int, default=1,
                        help="Number of training updates per one environment step, default = 1")
    # th 20210724
    parser.add_argument("-target_entropy", type=float, default=None, help="target entropy , default=Num action")
    # for MBPO and redq setting, Hopper: -1, HC: -3, Walker: -3, Ant: -4, Humaoid: -2
    #
    parser.add_argument("-method", default="sac", choices=["sac", "redq", "duvn", "monosac"], help="method, default=sac")
    # learning per steps
    parser.add_argument("-batch_size", type=int, default=256,
                        help="Number of training batch, default = 256")
    #
    parser.add_argument("-target_drop_rate", type=float, default=0.0, help="drop out rate of target value function, default=0")
    #
    parser.add_argument("-critic_update_delay", type=int, default=1, help="number of critic learning delay (tau and UDP is rescaled), default=1 (no delay)") # TODO remove

    # 20210813
    # dist RL added @ 20210711
    parser.add_argument("-profile", type=int, default=0, choices=[0, 1],
                        help="Using profile for cpu/gpu speed and memory usage if set to 1, default=0")


    args = parser.parse_args()

    # You can define configs in the external json or yaml file.
    configs = {
        'num_steps': args.frames, #3000000,
        'batch_size': args.batch_size, #, 256,
        'lr': 0.0003,
        'hidden_units': [256, 256],
        'memory_size': 1e6,
        'gamma': 0.99,
        'tau': 0.005,
        'entropy_tuning': True,
        'ent_coef': 0.2,  # It's ignored when entropy_tuning=True.
        'multi_step': args.n_step, #1,
        'per': args.per, #False,  # prioritized experience replay
        'alpha': 0.6,  # It's ignored when per=False.
        'beta': 0.4,  # It's ignored when per=False.
        'beta_annealing': (1.0 - 0.4) / (1.0 * args.frames * args.updates_per_step), # 0.0001,  # It's ignored when per=False.
        'grad_clip': None,
        'updates_per_step': args.updates_per_step * args.critic_update_delay, # args.updates_per_step, #1,
        'start_steps': 5000, #10000,
        'log_interval': 10,
        'target_update_interval': 1,
        'eval_interval': args.eval_every, # 10000,
        'cuda': args.gpu_id, # args.cuda,
        'seed': args.seed,

        # adde by TH
        'eval_runs': args.eval_runs,
        'huber': args.huber, # TODO remove
        'layer_norm': args.layer_norm,
        #
        'target_entropy': args.target_entropy,
        'method': args.method,
        'target_drop_rate': args.target_drop_rate,
        'critic_update_delay': args.critic_update_delay
    }

    env = gym.make(args.env)

    # make sparse en: TH 20210705
    if args.sparsity_th > 0.0 :
        print("Evaluation in sparse reward setting with lambda = " + str(args.sparsity_th))
        env = SparseRewardEnv(env, rew_thresh=args.sparsity_th)
        env._max_episode_steps = env.wrapped_env._max_episode_steps

    label = args.env + "_" + str(datetime.datetime.now().isoformat())
    log_dir = os.path.join('runs', args.info, label)

    if args.distributional: # TODO remove
        raise NotImplementedError()
        #print(" Use IQN agent")
        #agent = IQNSacAgent(env=env, log_dir=log_dir, **configs)
    else:
        if args.profile:
            agent = SacAgent4Profile(env=env, log_dir=log_dir, **configs)
        else:
            agent = SacAgent(env=env, log_dir=log_dir, **configs)
    agent.run()


if __name__ == '__main__':
    run()
