import minari
import os
import tensorflow as tf
from stable_baselines.ppo2 import PPO2


def load_environment_variables():
  """
  Load environment variables and convert them to the correct data type.
  """
  env_vars = {
      'ROOT_DIR': str,
      'SEED': int,
      'TOTAL_ENV_STEPS': int,
      'MODE': str,
      'VISUALIZE': lambda x: x.lower() in ['true', '1', 'yes'],
      'INT_SAVE_FREQ': int,
      'DATASET_ID': str,
      'BATCH_SIZE': int,
      'NUM_EPOCHS': int,
      'LEARNING_RATE': float
  }

  return {var: env_vars[var](os.environ[var]) for var in env_vars}


def episode_generator(dataset):
  """
  Generator for episodes from the dataset.
  """
  for episode in dataset:
    step_data = (episode.observations, episode.actions)
    for state, action in zip(*step_data):
      yield state, action


def prepare_dataset(dataset, buffer_size, batch_size):
  """
  Prepare the dataset for training, ensuring data loading happens on the CPU.
  """
  tf_dataset = tf.data.Dataset.from_generator(
      lambda: episode_generator(dataset),
      output_types=(tf.float32, tf.float32),
      output_shapes=(
          dataset.spec.observation_space.shape,
          dataset.spec.action_space.shape,
      ),
  )
  return tf_dataset.shuffle(buffer_size).batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)


def train():
  env_vars = load_environment_variables()

  print('Environment Variables:', env_vars)

  tf.random.set_seed(env_vars['SEED'])

  # Loading dataset
  dataset = minari.load_dataset(dataset_id=env_vars['DATASET_ID'],
                                download=False)
  buffer_size = 1000

  model = PPO2(
      'MlpPolicy',
      'QuadrupedLocomotion-v0',
      learning_rate=env_vars['LEARNING_RATE'],
      policy_kwargs=dict(act_fun=tf.nn.relu, layers=[512, 256]),
      verbose=1,
  )
  with model.sess.graph.as_default():
    tf_dataset = prepare_dataset(dataset, buffer_size, env_vars['BATCH_SIZE'])
    model.sess.run(tf.compat.v1.global_variables_initializer())
    model.pretrain(tf_dataset,
                   n_epochs=env_vars['NUM_EPOCHS'],
                   learning_rate=env_vars['LEARNING_RATE'],
                   adam_epsilon=1e-8,
                   val_interval=1,
                   summary_dir=os.path.join(env_vars['ROOT_DIR'],
                                            'summaries'))

  model.save(os.path.join(env_vars['ROOT_DIR'], 'final_bc_policy'))


if __name__ == "__main__":
  train()
