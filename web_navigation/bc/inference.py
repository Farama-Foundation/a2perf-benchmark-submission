import os

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from train import DQNLSTM

from rl_perf.domains.web_nav.CoDE import vocabulary_node


def load_model():
    pass

def preprocess_observation(observation):
    # Write your code here to preprocess the observation. This function should return the preprocessed observation.
    for key in observation:
        observation[key] = tf.convert_to_tensor(observation[key], dtype=observation[key].dtype)

    time_step = ts.TimeStep(step_type=ts.StepType.FIRST, reward=0.0, discount=1.0, observation=observation)

    # Convert the single timestep into a batch of size 1
    time_step = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), time_step)

    return time_step


def infer_once(model, observation):
    action_step = model.action(time_step=observation)
    action = tf.nest.map_structure(lambda t: tf.squeeze(t, axis=0), action_step.action)
    return action
