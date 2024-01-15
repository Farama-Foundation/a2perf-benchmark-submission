import os

import gymnasium as gym
import numpy as np
from tf_agents.policies import policy_loader
from tf_agents.trajectories import time_step as ts


def load_policy():
  root_dir = os.environ.get('ROOT_DIR', None)
  if root_dir is None:
    raise ValueError(
        'ROOT_DIR environment variable must be set to load the model.')

  saved_model_path = os.path.join(root_dir, 'policies', 'policy')
  checkpoint_path = os.path.join(root_dir, 'policies', 'checkpoints')

  # Get max checkpoint from checkpoint_path
  max_checkpoint = sorted(os.listdir(checkpoint_path))[-1]

  policy = policy_loader.load(saved_model_path=saved_model_path,
                              checkpoint_path=os.path.join(checkpoint_path,
                                                           max_checkpoint), )

  return policy


def preprocess_observation(observation, reward=0.0, discount=1.0,
    step_type=ts.StepType.MID):
  """Preprocess raw observation from Gym environment into TF Agents TimeStep."""
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


def infer_once(policy, preprocessed_observation):
  """Run a single inference step using the given policy."""
  action_step = policy.action(preprocessed_observation)
  return action_step.action


if __name__ == '__main__':

  # noinspection PyUnresolvedReferences
  from a2perf.domains import quadruped_locomotion

  env = gym.make('QuadrupedLocomotion-v0', motion_files=[
      "/home/ikechukwuu/workspace/rl-perf/a2perf/domains/quadruped_locomotion/motion_imitation/data/motions/dog_pace.txt"
  ], num_parallel_envs=1)
  policy = load_policy()

  # Run inference for a single episode
  obs, info = env.reset()
  terminated = False
  truncated = False
  while not (terminated or truncated):
    preprocessed_obs = preprocess_observation(obs)
    action = infer_once(policy, preprocessed_obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
