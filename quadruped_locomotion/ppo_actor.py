import os
import inspect

import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import tensorflow as tf
import time
import gym
import gin

from rl_perf.domains.quadruped_locomotion.motion_imitation.learning import imitation_policies as imitation_policies
from rl_perf.domains.quadruped_locomotion.motion_imitation.learning import ppo_imitation as ppo_imitation

from stable_baselines.common.callbacks import CheckpointCallback

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

ENABLE_ENV_RANDOMIZER = True


def set_rand_seed(seed=None):
    if seed is not None:
        seed = int(time.time())
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def train(
        motion_file_path,
        seed=None,
        mode=None,
        visualize=False,
        output_dir=None,
        optim_batchsize=0,
        timesteps_per_actorbatch=0,
        total_timesteps=0,
        int_save_freq=0):
    rank = MPI.COMM_WORLD.Get_rank()
    set_rand_seed(seed * rank)
    env = gym.make('QuadrupedLocomotionEnv-v0',
                   motion_files=motion_files,
                   mode=mode,
                   enable_rendering=visualize)

    policy_kwargs = {
        "net_arch": [{"pi": [512, 256],
                      "vf": [512, 256]}],
        "act_fun": tf.nn.relu
    }

    if rank == 0:
        callbacks.append(CheckpointCallback(save_freq=int_save_freq,
                                            save_path=policy_save_path,
                                            name_prefix='rl_model'))

    model = ppo_imitation.PPOImitation(
        policy=imitation_policies.ImitationPolicy,
        env=env,
        gamma=0.95,
        timesteps_per_actorbatch=timesteps_per_actorbatch,
        clip_param=0.2,
        optim_epochs=1,
        optim_stepsize=1e-5,
        optim_batchsize=optim_batchsize,
        lam=0.95,
        adam_epsilon=1e-5,
        schedule='constant',
        policy_kwargs=policy_kwargs,
        tensorboard_log=output_dir if rank == 0 else None,
        verbose=2 * (rank == 0),

    )

    model.learn(total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name="PPO")

    if rank == 0:
        model.save("final_ppo_policy")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
    arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None)
    arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
    arg_parser.add_argument("--motion_file_path", dest="motion_file_path", type=str, default=None)
    arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
    arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int,
                            default=0)  # save intermediate model every n policy steps

    args = arg_parser.parse_args()

    print("args.seed:", args.seed)
    print("args.mode:", args.mode)
    print("int_save_freq:", args.int_save_freq)
    print("args.output_dir:", args.output_dir)
    print("args.total_timesteps:", args.total_timesteps)
    print("args.motion_file_path:", args.motion_file_path)
    print("args.model_file:", args.model_file)
    print("args.visualize:", args.visualize)

    train(motion_file_path=args.motion_file_path,
          total_timesteps=args.total_timesteps,
          output_dir=args.output_dir,
          visualize=args.visualize,
          mode=args.mode,
          seed=args.seed,
          int_save_freq=args.int_save_freq)
