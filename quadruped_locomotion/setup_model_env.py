import inspect
import os.path

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import argparse
import logging

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.noise import NormalActionNoise
from stable_baselines.ddpg.policies import MlpPolicy

import gym
from rl_perf.domains.quadruped_locomotion.motion_imitation.learning.ddpg_imitation import DDPGImitation

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256
ENABLE_ENV_RANDOMIZER = True


def train(motion_file_path,
          int_save_freq=10,
          seed=0,
          total_timesteps=2e8,
          output_dir="output"):
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    rank = MPI.COMM_WORLD.Get_rank()
    parallel_cores = int(os.environ['PARALLEL_CORES'])
    policy_save_path = os.path.join(output_dir, 'policies')
    timesteps_per_actorbatch = int(np.ceil(float(TIMESTEPS_PER_ACTORBATCH) / parallel_cores))
    optim_batchsize = int(np.ceil(float(OPTIM_BATCHSIZE) / parallel_cores))

    env = gym.make('QuadrupedLocomotionEnv-v0', motion_files=[motion_file_path], mode='train', enable_rendering=False)
    eval_env = None

    if rank == 0:
        logging.info('Testing not creating eval env for rank %d', rank)
    else:
        logging.info("eval_env is not created for rank %d", rank)

    param_noise = None
    action_noise = None
    callbacks = []

    print("timesteps_per_actorbatch:", timesteps_per_actorbatch)
    print("optim_batchsize:", optim_batchsize)
    print("rank:", rank)
    print("parallel_cores:", parallel_cores)
    print("observation_space high:", env.observation_space.high)
    print("observation_space low:", env.observation_space.low)
    print("action_space high:", env.action_space.high)
    print("action_space low:", env.action_space.low)
    print('observation_space highest high:', np.max(env.observation_space.high))
    print('observation_space lowest low:', np.min(env.observation_space.low))
    print('action_space highest high:', np.max(env.action_space.high))
    print('action_space lowest low:', np.min(env.action_space.low))

    if rank == 0:
        callbacks.append(CheckpointCallback(save_freq=int_save_freq,
                                            save_path=policy_save_path,
                                            name_prefix='rl_model'))
    log_interval = -1
    logging.info("log_interval: %d", log_interval)

    model = DDPGImitation(policy=MlpPolicy,
                          env=env,
                          seed=seed,
                          policy_kwargs=dict(act_fun=tf.nn.relu,
                                             layers=[512, 256],
                                             layer_norm=True),
                          eval_env=eval_env,
                          batch_size=OPTIM_BATCHSIZE,
                          buffer_size=int(1e6),
                          critic_lr=1e-4,
                          actor_lr=1e-4,
                          tau=0.005,
                          observation_range=(-5, 5),
                          normalize_observations=True,
                          normalize_returns=False,
                          nb_train_steps=100,
                          nb_rollout_steps=100,
                          random_exploration=0.05,
                          verbose=2 * (rank == 0),
                          full_tensorboard_log=rank == 0,
                          tensorboard_log=output_dir if rank == 0 else None,
                          param_noise=param_noise,
                          action_noise=action_noise)
    model.learn(total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=log_interval,
                tb_log_name="DDPG")

    if rank == 0:
        model.save("final_ddpg_policy")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=0)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
    arg_parser.add_argument("--motion_file_path", dest="motion_file_path", type=str,
                            default="/rl-perf/rl_perf/domains/quadruped_locomotion/motion_imitation/data/motions/dog_pace.txt")
    arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
    arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int,
                            default=100)  # save intermediate model every n policy steps

    args = arg_parser.parse_args()

    print("args.seed:", args.seed)
    print("args.mode:", args.mode)
    print("int_save_freq:", args.int_save_freq)
    print("args.output_dir:", args.output_dir)
    print("args.total_timesteps:", args.total_timesteps)
    print("args.motion_file_path:", args.motion_file_path)
    print("args.visualize:", args.visualize)

    train(motion_file_path=args.motion_file_path,
          total_timesteps=args.total_timesteps,
          output_dir=args.output_dir,
          seed=args.seed,
          int_save_freq=args.int_save_freq)
