"""This script serves as an example inference script for participants in the A2Perf benchmark for autonomous agents.

It demonstrates essential steps in the inference process, including loading a
policy model, preprocessing observations from the environment, and executing
inference using the loaded policy.

Functions:
- load_policy: Loads a policy model based on the environment settings.
- preprocess_observation: Transforms raw observations from the Gym environment
into a format compatible with the policy.
- infer_once: Conducts a single inference step using the provided policy.
"""

import os
from collections import OrderedDict
from typing import Any
from typing import Union

import numpy as np
import tensorflow as tf
from absl import logging
from tf_agents.policies import policy_loader
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories import time_step as ts


def load_policy(env: Any) -> TFPolicy:
  """Loads a policy model from the environment's root directory.

  Args:
      env: The environment for which the policy is to be loaded.

  Returns:
      The loaded policy.

  Raises:
      ValueError: If the ROOT_DIR environment variable is not set.
  """
  root_dir = os.environ.get('ROOT_DIR', None)
  policy_name = os.environ.get('POLICY_NAME', None)

  if root_dir is None:
    raise ValueError(
        'ROOT_DIR environment variable must be set to load the model for inference.'
    )
  logging.info('Loading model from %s', root_dir)

  saved_model_path = os.path.join(root_dir, 'policies', policy_name)
  checkpoint_path = os.path.join(root_dir, 'policies', 'checkpoints')

  # Get max checkpoint from checkpoint_path
  max_checkpoint = sorted(os.listdir(checkpoint_path))[-1]
  logging.info('Loading checkpoint %s', max_checkpoint)

  policy = policy_loader.load(
      saved_model_path=saved_model_path,
      checkpoint_path=os.path.join(checkpoint_path, max_checkpoint),
  )
  logging.info('Successfully loaded policy')
  return policy


def infer_once(policy: TFPolicy, preprocessed_observation: ts.TimeStep) -> Any:
  """Runs a single inference step using the given policy.

  Args:
      policy: The policy to use for inference.
      preprocessed_observation: The preprocessed observation for inference.

  Returns:
      The action determined by the policy for the given observation.
  """
  action_step = policy.action(preprocessed_observation)
  return action_step.action


def preprocess_observation(
    observation: Union[np.ndarray, Any],
    reward: float = 0.0,
    discount: float = 1.0,
    step_type: ts.StepType = ts.StepType.MID,
    time_step_spec: ts.TimeStep = None,
) -> ts.TimeStep:
  """Preprocesses a raw observation from the Gym environment into a TF Agents TimeStep.

  Args:
      observation: Raw observation from the environment.
      reward: The reward received after the last action.
      discount: The discount factor.
      step_type: The type of the current step.
      time_step_spec: The spec of the time_step used to extract dtype and shape.

  Returns:
      A preprocessed TimeStep object suitable for the policy.
  """
  if isinstance(observation, (dict, OrderedDict)):
    processed_observation = {}
    for key, value in observation.items():
      if time_step_spec and key in time_step_spec.observation.keys():
        spec = time_step_spec.observation[key]
        # Adjust dtype and shape according to the time_step_spec
        processed_observation[key] = tf.convert_to_tensor(value,
                                                          dtype=spec.dtype)
      else:
        # Use the numpy dtype of the element that was passed in
        processed_observation[key] = tf.convert_to_tensor(value,
                                                          dtype=value.dtype)
    observation = processed_observation
  elif isinstance(observation, np.ndarray):
    # Use the time_step_spec to convert the ndarray to a tensor
    if time_step_spec:
      observation = tf.nest.map_structure(
          lambda spec, value: tf.convert_to_tensor(value, dtype=spec.dtype),
          time_step_spec.observation,
          observation,
      )
    else:
      # Convert the ndarray directly, using its own dtype
      observation = tf.convert_to_tensor(observation, dtype=observation.dtype)
  else:
    raise ValueError(
        'Observation type not recognized. Please provide an OrderedDict, dict, or NumPy array.'
    )

  # Convert step_type, reward, and discount using their respective dtypes from time_step_spec
  # if it is provided, otherwise default to the dtype inferred from the input
  step_type = tf.convert_to_tensor(step_type,
                                   dtype=time_step_spec.step_type.dtype if time_step_spec else step_type.dtype)
  reward = tf.convert_to_tensor(reward,
                                dtype=time_step_spec.reward.dtype if time_step_spec else np.float32)
  discount = tf.convert_to_tensor(discount,
                                  dtype=time_step_spec.discount.dtype if time_step_spec else np.float32)

  return ts.TimeStep(
      step_type=step_type,
      reward=reward,
      discount=discount,
      observation=observation,
  )
