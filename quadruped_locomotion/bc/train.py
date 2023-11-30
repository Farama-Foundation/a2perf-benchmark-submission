import os

import minari
import tensorflow as tf
from absl import app
from absl import flags
from stable_baselines import PPO2
from tensorflow.python.framework.ops import enable_eager_execution
import tensorflow.compat.v1 as tf1

from rl_perf.domains import quadruped_locomotion

# enable_eager_execution()
_MINARI_DATASETS_PATH = flags.DEFINE_string('minari_datasets_path', None,
                                            'Path to Minari datasets.')


def train():
  # turn on eager mode for debugging

  root_dir = os.environ['ROOT_DIR']
  seed = int(os.environ['SEED'])
  total_timesteps = int(os.environ['TOTAL_ENV_STEPS'])
  mode = os.environ['MODE']
  visualize = bool(os.environ['VISUALIZE'])
  int_save_freq = int(os.environ['INT_SAVE_FREQ'])
  motion_file_path = os.environ['MOTION_FILE_PATH']
  dataset_id = os.environ['DATASET_ID']
  output_dir = root_dir
  batch_size = int(os.environ['BATCH_SIZE'])

  if _MINARI_DATASETS_PATH.value is not None:
    os.environ['MINARI_DATASETS_PATH'] = _MINARI_DATASETS_PATH.value

  print("root_dir:", root_dir)
  print("seed:", seed)
  print("total_timesteps:", total_timesteps)
  print("mode:", mode)
  print("visualize:", visualize)
  print("int_save_freq:", int_save_freq)
  print("output_dir:", output_dir)
  print("batch_size:", batch_size)

  dataset = minari.load_dataset(dataset_id=dataset_id, download=False)

  import tensorflow as tf

  def episode_generator():
    for episode in dataset:
      step_data = (episode.observations, episode.actions)
      combined_step_data = zip(*step_data)
      for state, action in combined_step_data:
        yield state, action

  buffer_size = 1000
  model = PPO2('MlpPolicy',
               'QuadrupedLocomotion-v0',
               policy_kwargs=dict(act_fun=tf.nn.relu,
                                  layers=[512, 256]),
               verbose=1)

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
        )
    )

    tf_dataset = tf_dataset.shuffle(buffer_size).batch(batch_size).prefetch(
        tf1.data.experimental.AUTOTUNE)

    iterator = tf_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

  model.pretrain(next_element, n_epochs=1000)
  wxyz = 2


def main(_):
  train()


if __name__ == '__main__':
  app.run(main)
