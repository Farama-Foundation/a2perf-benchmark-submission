import argparse
import os.path

import numpy as np
from mpi4py import MPI
from stable_baselines import DDPG
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.ddpg.policies import MlpPolicy

from rl_perf.domains import quadruped_locomotion
import tensorflow as tf

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256
ENABLE_ENV_RANDOMIZER = True
def train(motion_file_path,
          int_save_freq=10,
          seed=0,
          total_timesteps=2e8,
          output_dir="output"):
    rank = MPI.COMM_WORLD.Get_rank()

    print(rank)
    env = quadruped_locomotion.motion_imitation.envs.env_builder.build_imitation_env(
        motion_files=[motion_file_path], enable_randomizer=ENABLE_ENV_RANDOMIZER,
        enable_rendering=False, mode='train')

    eval_env = quadruped_locomotion.motion_imitation.envs.env_builder.build_imitation_env(
        motion_files=[motion_file_path], enable_rendering=False, mode='test')
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    print(env.observation_space.high)
    print(env.observation_space.low)

    policy_save_path = os.path.join(output_dir, 'policies')
    print(policy_save_path)

    callbacks = []
    callbacks.append(CheckpointCallback(save_freq=int_save_freq,
                                        save_path=policy_save_path,
                                        name_prefix='rl_model'))

    model = DDPG(policy=MlpPolicy,
                 env=env,
                 seed=seed,
                 policy_kwargs=dict(act_fun=tf.nn.relu,
                                    layers=[512, 256]),
                 eval_env=eval_env,
                 batch_size=256,
                 buffer_size=int(1e6),
                 normalize_observations=False,
                 nb_eval_steps=200,
                 nb_train_steps=100,
                 nb_rollout_steps=100,
                 verbose=2,
                 tensorboard_log=output_dir,
                 n_cpu_tf_sess=None,
                 param_noise=param_noise,
                 action_noise=action_noise)

    model.learn(total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=1,  # we want to log after every single iteration of the training/eval loop
                tb_log_name="DDPG")

    if rank == 0:
        model.save("ddpg_test_out")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
    arg_parser.add_argument("--motion_file_path", dest="motion_file_path", type=str, default="")
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
    print("args.visualize:", args.visualize)

    train(motion_file_path=args.motion_file_path,
          total_timesteps=args.total_timesteps,
          output_dir=args.output_dir,
          seed=args.seed,
          int_save_freq=args.int_save_freq)
