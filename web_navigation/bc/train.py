import json
import multiprocessing
import os
import random
import time

import gymnasium as gym
import minari
import numpy as np
import tensorflow as tf
import tf_agents
from absl import app
from absl import logging
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import actor_policy
from tf_agents.policies import policy_saver
from tf_agents.utils import common

from a2perf.domains.web_navigation.gwob.CoDE import networks
from a2perf.domains.web_navigation.gwob.CoDE import vocabulary_node


def episode_generator(dataset):
  for episode in dataset:
    step_data = (episode.observations, episode.actions)
    combined_step_data = zip(*step_data)
    for state, action in combined_step_data:
      yield state, action


def create_env(env_seed: int, env_name='CartPole-v0', difficulty=None,
    global_vocab=None, env_args=None):
  return suite_gym.load(
      environment_name=env_name,
      spec_dtype_map={gym.spaces.Discrete: np.int32},
      gym_kwargs={
          'difficulty': difficulty,
          'seed': env_seed,
          'global_vocabulary': global_vocab,
          **env_args,
      },
  )


def train(_):
  # Using default values for environment variables
  batch_size = int(os.environ.get('BATCH_SIZE', '32'))
  difficulty_level = int(os.environ.get('DIFFICULTY_LEVEL', '1'))
  eval_interval = int(os.environ.get('EVAL_INTERVAL', '100'))
  learning_rate = float(os.environ.get('LEARNING_RATE', '0.001'))
  log_interval = int(os.environ.get('LOG_INTERVAL', '10'))
  num_epochs = int(os.environ.get('NUM_EPOCHS', '1'))
  policy_checkpoint_interval = int(
      os.environ.get('POLICY_CHECKPOINT_INTERVAL', '500'))
  root_dir = os.environ.get('ROOT_DIR', './root_dir')
  seed = int(os.environ.get('SEED', '42'))
  summary_interval = int(os.environ.get('SUMMARY_INTERVAL', '100'))
  train_checkpoint_interval = int(
      os.environ.get('TRAIN_CHECKPOINT_INTERVAL', '100'))
  use_tf_functions = os.environ.get('USE_TF_FUNCTIONS', 'False') == 'True'
  vocab_path = os.environ.get('VOCAB_PATH', './vocab_path')
  dataset_id = os.environ.get('DATASET_ID', None)
  num_eval_episodes = int(os.environ.get('NUM_EVAL_EPISODES', '10'))
  max_vocab_size = int(os.environ.get('MAX_VOCAB_SIZE', '500'))

  print(f'batch_size: {batch_size}')
  print(f'num_epochs: {num_epochs}')
  print(f'dataset_id: {dataset_id}')
  print(f'summary_interval: {summary_interval}')
  print(f'difficulty_level: {difficulty_level}')
  print(f'eval_interval: {eval_interval}')
  print(f'learning_rate: {learning_rate}')
  print(f'log_interval: {log_interval}')
  print(f'policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'root_dir: {root_dir}')
  print(f'seed: {seed}')
  print(f'train_checkpoint_interval: {train_checkpoint_interval}')
  tf.random.set_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  summary_dir = os.path.join(root_dir, 'summaries')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      os.path.join(summary_dir, 'train'))
  train_summary_writer.set_as_default()
  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      os.path.join(summary_dir, 'eval'))

  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]

  global_step = tf.Variable(0, trainable=False, name="global_step",
                            dtype=tf.int64)
  manager = multiprocessing.Manager()
  lock = manager.Lock()

  # Load the global vocabulary
  global_vocab = vocabulary_node.LockedMultiprocessingVocabulary(
      max_vocabulary_size=max_vocab_size,
      multiprocessing_lock=lock)
  with open(vocab_path, 'rb') as f:
    vocab = json.load(f)
    global_vocab.restore(dict(global_vocab=vocab))

  env_args = {'designs': [
      {'number_of_pages': 1, 'action': [], 'action_page': [], }],
      'cyclic_action_penalty': 0.0,
      'timestep_penalty': 0.0,
  }
  eval_env = tf_agents.environments.ParallelPyEnvironment(
      [lambda: create_env(seed, env_name='WebNavigation-v0',
                          difficulty=difficulty_level,
                          global_vocab=global_vocab,
                          env_args=env_args)])
  eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)

  time_step_spec = eval_tf_env.time_step_spec()
  observation_spec = eval_tf_env.observation_spec()
  action_spec = eval_tf_env.action_spec()

  with tf.name_scope('ActorNetwork'):
    actor_net = networks.WebLSTMActorDistributionNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        name='actor',
        lstm_kwargs=dict(
            vocab_size=max_vocab_size
            if max_vocab_size is not None
            else eval_tf_env.pyenv.envs[0].env.local_vocab.max_vocabulary_size,
            latent_dim=50,
            profile_value_dropout=0.0,
            embedding_dim=100, )
    )

  # Create the behavioral cloning agent
  with tf.name_scope('BCAgent'):
    tf_agent = behavioral_cloning_agent.BehavioralCloningAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        cloning_network=actor_net,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        num_outer_dims=1,
        epsilon_greedy=0,
        loss_fn=tf.keras.losses.CategoricalCrossentropy(),
        gradient_clipping=None,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        train_step_counter=global_step,
    )
  tf_agent.initialize()

  eval_policy = actor_policy.ActorPolicy(tf_agent.policy, clip=False)
  train_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(train_dir, 'train'),
      agent=tf_agent,
      global_step=global_step,
  )
  saved_model = policy_saver.PolicySaver(eval_policy, train_step=global_step)
  saved_model_dir = os.path.join(root_dir, 'policies')
  train_checkpointer.initialize_or_restore()

  if use_tf_functions:
    tf_agent.train = common.function(tf_agent.train)

  # Compute eval metrics once at the beginning of training
  metric_utils.eager_compute(
      eval_metrics,
      eval_tf_env,
      eval_policy,
      num_episodes=num_eval_episodes,
      train_step=global_step,
      summary_writer=eval_summary_writer,
      summary_prefix='Metrics',
  )
  metric_utils.log_metrics(eval_metrics)

  minari_dataset = minari.load_dataset(dataset_id=dataset_id, download=False)
  tf_dataset = tf.data.Dataset.from_generator(
      generator=lambda: episode_generator(minari_dataset),
      output_signature=(
          minari_dataset.spec.observation_space.shape,
          minari_dataset.spec.action_space.shape,
      )
  )
  tf_dataset = tf_dataset.shuffle(1000).batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE).repeat(num_epochs)
  iterator = iter(tf_dataset)

  def train_step():
    experience, _ = next(iterator)
    return tf_agent.train(experience)

  if use_tf_functions:
    train_step = common.function(train_step)

  time_acc = 0
  timed_at_step = global_step.value()
  iters_so_far = 0

  try:
    while True:
      start_time = time.time()
      train_time = time.time()
      train_loss = train_step()
      train_time = time.time() - train_time

      tf.summary.scalar('iters_so_far', iters_so_far, step=iters_so_far)
      global_step_val = global_step.value()

      if iters_so_far % log_interval == 0:
        time_acc += time.time() - start_time
        logging.info('step = %d, loss = %f', global_step_val.numpy(),
                     train_loss.loss)
        print('step = %d, loss = %f', global_step_val.numpy(), train_loss.loss)
        steps_per_sec = (
                            global_step_val.numpy() - timed_at_step.numpy()) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        print('%.3f steps/sec', steps_per_sec)
        tf.summary.scalar(name='global_steps_per_sec', data=steps_per_sec,
                          step=iters_so_far
                          )
        print(f'train_time: {train_time}')

        timed_at_step = global_step_val
        time_acc = 0

      if iters_so_far % eval_interval == 0:
        metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
            use_function=False,
        )
        metric_utils.log_metrics(eval_metrics)

      if iters_so_far % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)
        tokens = global_vocab.save()

      if iters_so_far % policy_checkpoint_interval == 0:
        # Use global step value for checkpoint directory name
        save_location = os.path.join(saved_model_dir, 'policy_' +
                                     str(global_step_val))
        saved_model.save(save_location)

      iters_so_far += 1
  except StopIteration:
    pass

  manager.shutdown()
  eval_tf_env.close()


if __name__ == '__main__':
  app.run(train)
