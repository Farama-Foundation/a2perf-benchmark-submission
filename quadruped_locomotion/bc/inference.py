import os

from absl import app
from stable_baselines import PPO2


def load_model(env):
  root_dir = os.environ['ROOT_DIR']
  seed = int(os.environ['SEED'])
  print("root_dir:", root_dir)
  print("seed:", seed)
  policy_path = os.path.join(root_dir, f'final_bc_policy.zip')
  print("policy_path:", policy_path)
  model = PPO2.load(policy_path, env=env)
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
