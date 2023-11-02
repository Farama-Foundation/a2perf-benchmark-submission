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


def load_model(env):
    root_dir = os.environ['ROOT_DIR']
    seed = int(os.environ['SEED'])
    print("root_dir:", root_dir)
    print("seed:", seed)

    policies_dir = os.path.join(root_dir, 'policies')
    policy_filename = f'rl_policy_9991750_steps.zip'
    policy_path = os.path.join(policies_dir, policy_filename)

    timesteps_per_actorbatch = 4096
    rank = 1
    optim_batchsize = 256

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
                          param_noise=None,
                          action_noise=None)
    model.load(policy_path)
    return model


def infer_once(model, observation):
    # the model is from stable baselines so use it to run inference on a single observation
    action, _states = model.predict(observation)
    return action


def preprocess_observation(observation):
    return observation


def main(_):
    pass


if __name__ == '__main__':
    app.run(main)
