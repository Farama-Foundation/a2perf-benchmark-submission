"""
This script serves as an example inference script for participants in the A2Perf benchmark for autonomous agents.
It demonstrates essential steps in the inference process, including loading a policy model, preprocessing observations
from the environment, and executing inference using the loaded policy.

Functions:
- load_policy: Loads a policy model based on the environment settings.
- preprocess_observation: Transforms raw observations from the Gym environment into a format compatible with the policy.
- infer_once: Conducts a single inference step using the provided policy.
"""

import os
from typing import Any
from typing import Union

import numpy as np
from absl import logging
from tf_agents.policies import policy_loader
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories import time_step as ts


def load_policy(env: Any) -> TFPolicy:
  """
  Loads a policy model from the environment's root directory.

  Args:
      env: The environment for which the policy is to be loaded.

  Returns:
      The loaded policy.

  Raises:
      ValueError: If the ROOT_DIR environment variable is not set.
  """
  root_dir = os.environ.get('ROOT_DIR', None)
  if root_dir is None:
    raise ValueError(
        'ROOT_DIR environment variable must be set to load the model.')
  logging.info('Loading model from %s', root_dir)

  saved_model_path = os.path.join(root_dir, 'policies', 'policy')
  checkpoint_path = os.path.join(root_dir, 'policies', 'checkpoints')

  # Get max checkpoint from checkpoint_path
  max_checkpoint = sorted(os.listdir(checkpoint_path))[-1]
  logging.info('Loading checkpoint %s', max_checkpoint)

  policy = policy_loader.load(saved_model_path=saved_model_path,
                              checkpoint_path=os.path.join(checkpoint_path,
                                                           max_checkpoint), )
  logging.info('Successfully loaded policy')
  return policy


def preprocess_observation(observation: Union[np.ndarray, list],
    reward: float = 0.0,
    discount: float = 1.0,
    step_type: ts.StepType = ts.StepType.MID) -> ts.TimeStep:
  """
  Preprocesses a raw observation from the Gym environment into a TF Agents TimeStep.

  Args:
      observation: Raw observation from the environment.
      reward: The reward received after the last action.
      discount: The discount factor.
      step_type: The type of the current step.

  Returns:
      A preprocessed TimeStep object suitable for the policy.
  """
  # Ensure observation is a 1-D array
  observation = np.array(observation, dtype=np.float32)

  # Convert step_type to a numpy int32
  step_type = np.array(step_type, dtype=np.int32)

  return ts.TimeStep(
      step_type=step_type,  # Step type as numpy int32
      reward=np.float32(reward),  # Reward as single float32 value
      discount=np.float32(discount),  # Discount as single float32 value
      observation=observation  # Observation as 1-D array
  )


def infer_once(policy: TFPolicy, preprocessed_observation: ts.TimeStep) -> Any:
  """
  Runs a single inference step using the given policy.

  Args:
      policy: The policy to use for inference.
      preprocessed_observation: The preprocessed observation for inference.

  Returns:
      The action determined by the policy for the given observation.
  """
  action_step = policy.action(preprocessed_observation)
  return action_step.action
