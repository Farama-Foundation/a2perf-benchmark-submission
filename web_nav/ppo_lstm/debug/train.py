# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Train and Eval PPO.

To run:

```bash
tensorboard --logdir $HOME/tmp/ppo/gym/HalfCheetah-v2/ --port 2223 &

python tf_agents/agents/ppo/examples/v2/train_eval_clip_agent.py \
  --root_dir=$HOME/tmp/ppo/gym/HalfCheetah-v2/ \
  --logtostderr
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow_probability as tfp
import functools
import multiprocessing as mp
import os
import time

import gin
import gymnasium as gym
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tf_agents
from absl import app
from absl import logging
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import network
from tf_agents.networks import value_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from rl_perf.domains.web_nav.gwob.CoDE import networks
from rl_perf.domains.web_nav.gwob.CoDE import vocabulary_node


def remove_config_lines(config_string, key):
  lines = config_string.split('\n')
  lines = [line for line in lines if not line.startswith(key)]
  return '\n'.join(lines)


def create_env(env_seed: int, env_name='CartPole-v0', difficulty=None,
    global_vocab=None, env_args=None):
  return suite_gym.load(
      environment_name=env_name,
      spec_dtype_map={gym.spaces.Discrete: np.int32},
      gym_kwargs={
          'difficulty': difficulty,
          'seed': env_seed,
          'global_vocabulary': global_vocab,
          **env_args,  # Add env_args to the gym_kwargs dictionary
      },
  )


@gin.configurable
class PPOLSTMActor(network.DistributionNetwork):
  """Creates an actor producing either Normal or Categorical distribution.

  Note: By default, this network uses `NormalProjectionNetwork` for continuous
  projection which by default uses `tanh_squash_to_spec` to normalize its
  output. Due to the nature of the `tanh` function, values near the spec bounds
  cannot be returned.
  """

  def __init__(self,
      input_tensor_spec,
      output_tensor_spec,
      batch_squash=True,
      state_spec=(),
      discrete_projection_net=actor_distribution_network._categorical_projection_net,
      name='ActorDistributionNetwork', **kwargs):
    self._lstm = networks.WebLSTMActor(**kwargs)
    self._batch_squash = batch_squash

    def map_proj(spec):
      return discrete_projection_net(spec)

    projection_networks = tf.nest.map_structure(map_proj, output_tensor_spec)
    output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                        projection_networks)

    super(PPOLSTMActor, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=state_spec,
        output_spec=output_spec,
        name=name)

    self._projection_networks = projection_networks
    self._output_tensor_spec = output_tensor_spec

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self, inputs, *args, **kwargs):
    training = kwargs.get('training', False)
    network_state = kwargs.get('network_state', ())

    batch_squash = None
    outer_rank = None

    if self._batch_squash:
      outer_rank = tf_agents.utils.nest_utils.get_outer_rank(
          inputs, self.input_tensor_spec)
      batch_squash = tf_agents.networks.utils.BatchSquash(outer_rank)
      inputs = tf.nest.map_structure(batch_squash.flatten, inputs)

    state = self._lstm(observation=inputs,
                       is_training=training, )

    def call_projection_net(proj_net):
      if batch_squash is not None:
        unflattened_state = batch_squash.unflatten(state)
      else:
        unflattened_state = state

      distribution, _ = proj_net(
          unflattened_state, outer_rank, training=training)

      return distribution

    output_actions = tf.nest.map_structure(
        call_projection_net, self._projection_networks)
    return output_actions, network_state


class PPOLSTMValue(network.Network):
  """Feed Forward value network. Reduces to 1 value output per batch item."""

  def __init__(self,
      input_tensor_spec,
      batch_squash=True,
      state_spec=(),
      name='PPOLSTMValue', **kwargs):
    super(PPOLSTMValue, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=state_spec,
        name=name)
    self._batch_squash = batch_squash
    self._lstm = networks.WebLSTMActor(**kwargs)
    self._postprocessing_layers = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.random_uniform_initializer(
            minval=-0.03, maxval=0.03))

  def call(self, observation, step_type=None, network_state=(), training=False):
    batch_squash = None
    if self._batch_squash:
      outer_rank = tf_agents.utils.nest_utils.get_outer_rank(
          observation, self.input_tensor_spec)
      batch_squash = tf_agents.networks.utils.BatchSquash(outer_rank)
      observation = tf.nest.map_structure(batch_squash.flatten, observation)

    # Pass input through LSTM
    state = self._lstm(observation=observation,
                       is_training=training,
                       )

    # Get the value prediction for each observation
    output_value = tf.nest.map_structure(self._postprocessing_layers, state)
    output_value = tf.squeeze(output_value, -1)

    # After squashing the batch, we need to put the batch dimension back in
    if batch_squash is not None:
      output_value = tf.nest.map_structure(batch_squash.unflatten, output_value)
    return output_value, network_state


def create_env(env_seed: int, env_name='CartPole-v0', difficulty=None,
    global_vocab=None, env_args=None):
  return suite_gym.load(
      environment_name=env_name,
      spec_dtype_map={gym.spaces.Discrete: np.int32},
      gym_kwargs={
          'difficulty': difficulty,
          'seed': env_seed,
          'global_vocabulary': global_vocab,
          **env_args,  # Add env_args to the gym_kwargs dictionary
      },
  )


@gin.configurable
def train_eval(
    root_dir,
    env_name='WebNavigation-v0',
    difficulty=None,
    # Params for collect
    collect_steps_per_iteration=1,
    max_vocab_size=500,
    # Params for train
    eval_interval=1,
    total_env_steps=100000,
    batch_size=32,
    replay_buffer_capacity=1001,
    environment_batch_size=1,
    learning_rate=1e-4,
    gradient_clipping=None,
    use_tf_functions=False,
    # Params for checkpoints
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    # Params for summaries and logging
    num_eval_episodes=1,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    num_epochs=1,
    env_args=None,
    seed=0,
):
  """A simple train and eval for PPO."""
  if root_dir is None:
    raise AttributeError('train_eval requires a root_dir.')

  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')
  saved_model_dir = os.path.join(root_dir, 'policy_saved_model')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000
  )
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000
  )
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  manager = mp.Manager()
  lock = manager.Lock()
  global_vocab = (
      vocabulary_node.LockedVocabulary(max_vocabulary_size=max_vocab_size,
                                       multiprocessing_lock=lock)
  )
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)
  ):
    if seed is not None:
      tf.compat.v1.set_random_seed(seed)
    envs = [
        lambda: create_env(seed + i, env_name=env_name, difficulty=difficulty,
                           global_vocab=global_vocab,
                           env_args=env_args) for i
        in range(environment_batch_size)]
    eval_env = tf_agents.environments.ParallelPyEnvironment(
        [lambda: create_env(seed + environment_batch_size, env_name=env_name,
                            difficulty=difficulty,
                            global_vocab=global_vocab,
                            env_args=env_args)])
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)
    parallel_py_env = tf_agents.environments.ParallelPyEnvironment(envs,
                                                                   blocking=True,
                                                                   start_serially=True)
    tf_env = tf_py_environment.TFPyEnvironment(parallel_py_env)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    with tf.name_scope('Actor'):
      actor_net = PPOLSTMActor(
          input_tensor_spec=observation_spec,
          batch_squash=True,
          output_tensor_spec=action_spec,
          vocab_size=max_vocab_size
          if max_vocab_size is not None
          else tf_env.pyenv.envs[0].env.local_vocab.max_vocabulary_size,
          profile_value_dropout=0.0,
          embedding_dim=100,
          name='actor',
          latent_dim=50,
      )
      # extra_args = dict(fw_bs_encoder=actor_net._lstm._fw_bs_encoder if hasattr(
      #     actor_net._lstm, '_fw_bs_encoder') else None,
      #                   dom_encoder_bw=actor_net._lstm._dom_encoder_bw if hasattr(
      #                       actor_net._lstm, '_dom_encoder_bw') else None)
      # extra_args = {k: v for k, v in extra_args.items() if v is not None}

    # state spec must be defined so that it can be used to reshape our value predictions to [B, T, 1]
    with tf.name_scope('Value'):
      critic_net = PPOLSTMValue(
          input_tensor_spec=observation_spec,
          batch_squash=True,
          # state_spec=tf.TensorSpec([1]),
          state_spec=tf.TensorSpec([1]),
          vocab_size=max_vocab_size
          if max_vocab_size is not None
          else tf_env.pyenv.envs[
            0].env.local_vocab.max_vocabulary_size,
          profile_value_dropout=0.0,
          embedding_dim=100,
          name='value',
          latent_dim=50,
          # **extra_args
      )
    with tf.name_scope('PPOAgent'):
      tf_agent = ppo_clip_agent.PPOClipAgent(
          time_step_spec=time_step_spec,
          action_spec=action_spec,
          optimizer=optimizer,
          actor_net=actor_net,
          value_net=critic_net,
          entropy_regularization=0.0,
          importance_ratio_clipping=0.2,
          normalize_observations=False,
          normalize_rewards=False,
          use_gae=True,
          num_epochs=num_epochs,
          debug_summaries=debug_summaries,
          summarize_grads_and_vars=summarize_grads_and_vars,
          train_step_counter=global_step,
      )
    tf_agent.initialize()

    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]

    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(batch_size=environment_batch_size),
        tf_metrics.AverageEpisodeLengthMetric(
            batch_size=environment_batch_size
        ),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=environment_batch_size,
        max_length=replay_buffer_capacity,
    )

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
    )
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step,
    )
    saved_model = policy_saver.PolicySaver(eval_policy, train_step=global_step)

    train_checkpointer.initialize_or_restore()

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        env=tf_env,
        policy=collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=collect_steps_per_iteration,
    )

    def train_step():
      trajectories = replay_buffer.gather_all()
      return tf_agent.train(experience=trajectories)

    if use_tf_functions:
      # TODO(b/123828980): Enable once the cause for slowdown was identified.
      collect_driver.run = common.function(collect_driver.run, autograph=False)
      tf_agent.train = common.function(tf_agent.train, autograph=False)
      train_step = common.function(train_step)

    collect_time = 0
    train_time = 0
    timed_at_step = global_step.numpy()

    while environment_steps_metric.result() < total_env_steps:
      global_step_val = global_step.numpy()
      if global_step_val % eval_interval == 0:
        metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )

      start_time = time.time()
      collect_driver.run()
      collect_time += time.time() - start_time
      # tf_env.render()
      start_time = time.time()
      total_loss, ppo_loss_info = train_step()

      for loss in ['policy_gradient', 'value_estimation',
                   # 'l2_regularization',
                   # 'entropy_regularization',
                   # 'kl_penalty'
                   ]:
        attribute_name = f'{loss}_loss'
        print(f'{attribute_name}:  {getattr(ppo_loss_info, attribute_name)}')
      replay_buffer.clear()
      train_time += time.time() - start_time

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=step_metrics
        )

      if global_step_val % log_interval == 0:
        logging.info('step = %d, loss = %f', global_step_val, total_loss)
        steps_per_sec = (global_step_val - timed_at_step) / (
            collect_time + train_time
        )
        logging.info('%.3f steps/sec', steps_per_sec)
        logging.info(
            'collect_time = %.3f, train_time = %.3f', collect_time, train_time
        )
        with tf.compat.v2.summary.record_if(True):
          tf.compat.v2.summary.scalar(
              name='global_steps_per_sec', data=steps_per_sec, step=global_step
          )

        if global_step_val % train_checkpoint_interval == 0:
          train_checkpointer.save(global_step=global_step_val)

        if global_step_val % policy_checkpoint_interval == 0:
          policy_checkpointer.save(global_step=global_step_val)
          saved_model_path = os.path.join(
              saved_model_dir, 'policy_' + ('%d' % global_step_val).zfill(9)
          )
          saved_model.save(saved_model_path)

        timed_at_step = global_step_val
        collect_time = 0
        train_time = 0

    # One final eval before exiting.
    metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
    )


def train_mp(_):
  tf.compat.v1.enable_v2_behavior()

  # Extract environment variables
  seed = int(os.environ.get('SEED', None))
  root_dir = os.environ.get('ROOT_DIR', None)
  env_batch_size = int(os.environ.get('ENV_BATCH_SIZE', None))
  total_env_steps = int(os.environ.get('TOTAL_ENV_STEPS', None))
  difficulty_level = int(os.environ.get('DIFFICULTY_LEVEL', None))
  eval_interval = int(os.environ.get('EVAL_INTERVAL', None))
  train_checkpoint_interval = int(
      os.environ.get('TRAIN_CHECKPOINT_INTERVAL', None))
  policy_checkpoint_interval = int(
      os.environ.get('POLICY_CHECKPOINT_INTERVAL', None))
  rb_checkpoint_interval = int(os.environ.get('RB_CHECKPOINT_INTERVAL', None))
  rb_capacity = int(os.environ.get('RB_CAPACITY', None))
  log_interval = int(os.environ.get('LOG_INTERVAL', None))
  learning_rate = float(os.environ.get('LEARNING_RATE', None))
  summary_interval = int(os.environ.get('SUMMARY_INTERVAL', None))
  timesteps_per_actorbatch_param = int(
      os.environ.get('TIMESTEPS_PER_ACTORBATCH', None)
  )
  num_epochs = int(os.environ.get('NUM_EPOCHS', None))
  batched_total_env_steps = total_env_steps // env_batch_size
  timesteps_per_actorbatch = max(
      1, timesteps_per_actorbatch_param // env_batch_size
  )

  # Print extracted and computed values
  print(f'seed: {seed}')
  print(f'root_dir: {root_dir}')
  print(f'env_batch_size: {env_batch_size}')
  print(f'total_env_steps: {total_env_steps}')
  print(f'difficulty_level: {difficulty_level}')
  print(f'eval_interval: {eval_interval}')
  print(f'train_checkpoint_interval: {train_checkpoint_interval}')
  print(f'policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'rb_checkpoint_interval: {rb_checkpoint_interval}')
  print(f'log_interval: {log_interval}')
  print(f'summary_interval: {summary_interval}')
  print(f'learning_rate: {learning_rate}')

  # Convert all of the intervals to be in terms of iterations instead of environment steps
  eval_interval = max(1, eval_interval // timesteps_per_actorbatch_param)
  train_checkpoint_interval = max(
      1, train_checkpoint_interval // timesteps_per_actorbatch_param
  )
  policy_checkpoint_interval = max(
      1, policy_checkpoint_interval // timesteps_per_actorbatch_param
  )
  rb_checkpoint_interval = max(
      1, rb_checkpoint_interval // timesteps_per_actorbatch_param
  )
  log_interval = max(1, log_interval // timesteps_per_actorbatch_param)
  summary_interval = max(1, summary_interval // timesteps_per_actorbatch_param)

  print(f'eval_interval: {eval_interval}')
  print(f'train_checkpoint_interval: {train_checkpoint_interval}')
  print(f'policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'rb_checkpoint_interval: {rb_checkpoint_interval}')
  print(f'log_interval: {log_interval}')
  print(
      f'summary_interval: {summary_interval}'
  )  # Call train_eval function with required parameters
  train_eval(
      seed=seed,
      root_dir=root_dir,
      difficulty=difficulty_level,
      environment_batch_size=env_batch_size,
      learning_rate=learning_rate,
      replay_buffer_capacity=rb_capacity,
      num_eval_episodes=1,
      num_epochs=num_epochs,
      total_env_steps=batched_total_env_steps,
      collect_steps_per_iteration=timesteps_per_actorbatch,
      train_checkpoint_interval=train_checkpoint_interval,
      policy_checkpoint_interval=policy_checkpoint_interval,
      log_interval=log_interval,
      summary_interval=summary_interval,
      env_args={'designs': [
          {'number_of_pages': 1, 'action': [], 'action_page': [], }]}
  )


def train():
  tf_agents.system.multiprocessing.handle_main(train_mp)


if __name__ == '__main__':
  tf_agents.system.multiprocessing.handle_main(
      functools.partial(app.run, train_mp))
