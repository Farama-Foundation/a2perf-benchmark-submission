import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import tensorflow as tf
from mpi4py import MPI

from a2perf.domains.quadruped_locomotion.motion_imitation.learning import imitation_policies as imitation_policies
from a2perf.domains.quadruped_locomotion.motion_imitation.learning import ppo_imitation as ppo_imitation

ENABLE_ENV_RANDOMIZER = True


def set_rand_seed(seed=None):
    if seed is not None:
        seed = int(time.time())
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def train(
        motion_file_path,
        seed,
        mode,
        visualize,
        output_dir,
        optim_batchsize,
        timesteps_per_actorbatch,
        total_timesteps,
        int_save_freq):
    rank = MPI.COMM_WORLD.Get_rank()
    parallel_cores = MPI.COMM_WORLD.Get_size()
    set_rand_seed(seed * rank)
    env = gym.make('QuadrupedLocomotion-v0', motion_files=[motion_file_path], mode=mode, enable_rendering=visualize)

    policy_kwargs = {
        "net_arch": [{"pi": [512, 256],
                      "vf": [512, 256]}],
        "act_fun": tf.nn.relu
    }
    policy_save_path = os.path.join(output_dir, 'policies')
    os.makedirs(policy_save_path, exist_ok=True)
    callbacks = []
    tensorboard_log_dir = os.path.join(output_dir, 'tensorboard')
    model = ppo_imitation.PPOImitation(
        policy=imitation_policies.ImitationPolicy,
        env=env,
        gamma=0.99,
        timesteps_per_actorbatch=timesteps_per_actorbatch,
        clip_param=0.2,
        optim_epochs=1,
        optim_stepsize=1e-5,
        optim_batchsize=optim_batchsize,
        lam=0.95,
        adam_epsilon=1e-5,
        full_tensorboard_log=rank == 0,
        schedule='constant',
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log_dir if rank == 0 else None,
        verbose=2 * (rank == 0),

    )
    # Since one iteration corresponds with 4096 steps, we use this to compute the save frequency in terms of iterations
    save_iters = int(int_save_freq / (parallel_cores * timesteps_per_actorbatch))
    print("save_iters:", save_iters)
    print(f'save_iters corresponds to {save_iters * parallel_cores * timesteps_per_actorbatch} environment steps')
    model.learn(total_timesteps=total_timesteps,
                callback=callbacks,
                save_path=policy_save_path,
                save_iters=save_iters,
                tb_log_name=f'PPO_{str(rank)}')

    if rank == 0:
      model.save(os.path.join(output_dir, "final_ppo_policy"))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
    arg_parser.add_argument("--motion_file_path", dest="motion_file_path", type=str, default=None)
    arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
    arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int,
                            default=0)  # save intermediate model every n policy steps
    arg_parser.add_argument("--optim_batchsize", dest="optim_batchsize", type=int, default=0)
    arg_parser.add_argument("--timesteps_per_actorbatch", dest="timesteps_per_actorbatch", type=int,
                            default=0)

    args = arg_parser.parse_args()

    print("args.seed:", args.seed)
    print("args.mode:", args.mode)
    print("int_save_freq:", args.int_save_freq)
    print("args.output_dir:", args.output_dir)
    print("args.total_timesteps:", args.total_timesteps)
    print("args.motion_file_path:", args.motion_file_path)
    print("args.visualize:", args.visualize)
    print("args.int_save_freq:", args.int_save_freq)
    print("args.optim_batchsize:", args.optim_batchsize)
    print("args.timesteps_per_actorbatch:", args.timesteps_per_actorbatch)
    train(motion_file_path=args.motion_file_path,
          total_timesteps=args.total_timesteps,
          output_dir=args.output_dir,
          visualize=args.visualize,
          mode=args.mode,
          seed=args.seed,
          int_save_freq=args.int_save_freq,
          optim_batchsize=args.optim_batchsize,
          timesteps_per_actorbatch=args.timesteps_per_actorbatch)
