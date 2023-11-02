import argparse
import os.path
import random
import time

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines.ddpg.policies import MlpPolicy

from rl_perf.domains import quadruped_locomotion
from ddpg_imitation import DDPGImitation

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
        seed,
        mode,
        visualize,
        output_dir,
        optim_batchsize,
        timesteps_per_actorbatch,
        total_timesteps,
        int_save_freq,
        int_eval_freq
):
    rank = MPI.COMM_WORLD.Get_rank()
    parallel_cores = MPI.COMM_WORLD.Get_size()
    set_rand_seed(seed * rank)

    env = quadruped_locomotion.motion_imitation.envs.env_builder.build_imitation_env(
        motion_files=[motion_file_path], enable_randomizer=ENABLE_ENV_RANDOMIZER,
        enable_rendering=visualize, mode=mode)

    eval_env = quadruped_locomotion.motion_imitation.envs.env_builder.build_imitation_env(
        motion_files=[motion_file_path], enable_rendering=visualize, mode='test')

    n_actions = env.action_space.shape[-1]
    param_noise = None

    print(env.action_space.high)
    print(env.action_space.low)

    # this means that 67% of the time, the action will be within 0.1 of the mean
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))
    print(env.observation_space.high)
    print(env.observation_space.low)

    policy_save_path = os.path.join(output_dir, 'policies')
    os.makedirs(policy_save_path, exist_ok=True)

    # Tensorboard makes its own directory
    tensorboard_log_dir = os.path.join(output_dir, 'tensorboard')

    print(policy_save_path)

    # Ensure the results are not zero using the max function
    save_iters = max(1, int(int_save_freq / (parallel_cores * timesteps_per_actorbatch)))
    eval_iters = max(1, int(int_eval_freq / (parallel_cores * timesteps_per_actorbatch)))

    print("save_iters:", save_iters)
    print(f'save_iters corresponds to {save_iters * parallel_cores * timesteps_per_actorbatch} environment steps')

    print("eval_iters:", eval_iters)
    print(f'eval_iters corresponds to {eval_iters * parallel_cores * timesteps_per_actorbatch} environment steps')

    model = DDPGImitation(policy=MlpPolicy,
                          env=env,
                          seed=seed,
                          policy_kwargs=dict(act_fun=tf.nn.relu,
                                             layers=[512, 256]),
                          eval_env=eval_env,
                          buffer_size=int(1e6),
                          normalize_observations=False,
                          # normalize_returns=True,
                          normalize_returns=False,
                          tau=0.005,
                          nb_eval_episodes=1,
                          # batch_size=optim_batchsize,
                          batch_size=24,
                          actor_lr=1e-5,
                          critic_lr=1e-4,
                          adam_epsilon=1e-5,
                          random_exploration=0.0,
                          nb_train_steps=timesteps_per_actorbatch,
                          nb_rollout_steps=timesteps_per_actorbatch,
                          verbose=2 if rank == 0 else 0,
                          full_tensorboard_log=rank == 0,
                          tensorboard_log=tensorboard_log_dir if rank == 0 else None,
                          param_noise=param_noise,
                          action_noise=action_noise)

    # Since one iteration corresponds with 4096 steps, we use this to compute the save frequency in terms of iterations
    model.learn(total_timesteps=total_timesteps,
                save_path=policy_save_path,
                save_iters=save_iters,
                eval_iters=eval_iters,
                tb_log_name=f'DDPG_{str(rank)}')

    if rank == 0:
        model.save("final_ddpg_policy")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=0)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
    arg_parser.add_argument("--motion_file_path", dest="motion_file_path", type=str,
                            default='/rl-perf/rl_perf/domains/quadruped_locomotion/motion_imitation/data/motions/dog_pace.txt')
    arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
    arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int,
                            default=0)  # save intermediate model every n policy steps
    arg_parser.add_argument("--int_eval_freq", dest="int_eval_freq", type=int,
                            default=0)
    arg_parser.add_argument("--optim_batchsize", dest="optim_batchsize", type=int, default=0)
    arg_parser.add_argument("--timesteps_per_actorbatch", dest="timesteps_per_actorbatch", type=int,
                            default=0)

    args = arg_parser.parse_args()

    print("args.seed:", args.seed)
    print("args.mode:", args.mode)
    print("int_save_freq:", args.int_save_freq)
    print("int_eval_freq:", args.int_eval_freq)
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
          int_eval_freq=args.int_eval_freq,
          optim_batchsize=args.optim_batchsize,
          timesteps_per_actorbatch=args.timesteps_per_actorbatch)
