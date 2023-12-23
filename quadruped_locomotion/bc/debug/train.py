import minari
import os
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from absl import app

from stable_baselines import PPO2


def train():
  tf1.enable_v2_behavior()

  root_dir = os.environ['ROOT_DIR']
  seed = int(os.environ['SEED'])
  total_timesteps = int(os.environ['TOTAL_ENV_STEPS'])
  mode = os.environ['MODE']
  visualize = bool(os.environ['VISUALIZE'])
  int_save_freq = int(os.environ['INT_SAVE_FREQ'])
  dataset_id = os.environ['DATASET_ID']
  output_dir = root_dir
  summary_dir = os.path.join(output_dir, 'summaries')
  batch_size = int(os.environ['BATCH_SIZE'])
  num_epochs = int(os.environ['NUM_EPOCHS'])
  learning_rate = float(os.environ['LEARNING_RATE'])

  print('root_dir:', root_dir)
  print('seed:', seed)
  print('total_timesteps:', total_timesteps)
  print('mode:', mode)
  print('visualize:', visualize)
  print('int_save_freq:', int_save_freq)
  print('output_dir:', output_dir)
  print('batch_size:', batch_size)

  dataset = minari.load_dataset(dataset_id=dataset_id, download=False)

  def episode_generator():
    for episode in dataset:
      step_data = (episode.observations, episode.actions)
      combined_step_data = zip(*step_data)
      for state, action in combined_step_data:
        yield state, action

  buffer_size = 10000
  model = PPO2(
      'MlpPolicy',
      'QuadrupedLocomotion-v0',
      learning_rate=learning_rate,
      policy_kwargs=dict(act_fun=tf.nn.relu, layers=[512, 256]),
      verbose=1,
  )

  # Access the model's session and its graph
  sess = model.sess
  graph = sess.graph

  with graph.as_default():
    # Define the dataset within the graph
    tf_dataset = tf1.data.Dataset.from_generator(
        episode_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            dataset.spec.observation_space.shape,
            dataset.spec.action_space.shape,
        ),
    )

    tf_dataset = (
        tf_dataset.shuffle(buffer_size)
        .batch(batch_size)
        .prefetch(tf1.data.experimental.AUTOTUNE)
    )

    iterator = tf_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

  model.pretrain(next_element,
                 n_epochs=num_epochs,
                 learning_rate=learning_rate,
                 adam_epsilon=1e-8,
                 val_interval=1,
                 summary_dir=summary_dir,
                 )

  # Save the model
  model.save(os.path.join(output_dir, 'final_bc_policy'))


def main(_):
  train()


if __name__ == '__main__':
  app.run(main)
