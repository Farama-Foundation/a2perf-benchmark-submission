import argparse
import os.path
import random
import time
import functools

import numpy as np
import tensorflow as tf
from mpi4py import MPI
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines.ddpg.policies import MlpPolicy

from a2perf.domains import quadruped_locomotion
from ddpg_imitation import DDPGImitation


def load_model(env):
    root_dir = os.environ['ROOT_DIR']
    seed = int(os.environ['SEED'])
    print("root_dir:", root_dir)
    print("seed:", seed)

    policies_dir = os.path.join(root_dir, 'policies')
    max_policy = functools.reduce(max, [int(x.split('_')[2]) for x in os.listdir(policies_dir)])
    policy_path = os.path.join(policies_dir, f'rl_policy_{max_policy}_steps.zip')
    print("policy_path:", policy_path)

    model = DDPGImitation.load(policy_path, env=env)

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
