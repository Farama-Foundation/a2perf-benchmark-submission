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

import functools
import os
import random

import gin
import gymnasium as gym
import numpy as np
import reverb
import tensorflow as tf
import tf_agents
from absl import app
from absl import logging
from tf_agents import networks
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.environments import suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from a2perf.domains import quadruped_locomotion

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256


def create_env(
    env_name='CartPole-v0',
    env_args=None,
):
  return suite_pybullet.load(
      environment_name=env_name,
      spec_dtype_map={gym.spaces.Discrete: np.int32},
      gym_kwargs=env_args,
  )


@gin.configurable
def train_eval(
    root_dir,
    env_name='WebNavigation-v0',
    # Params for collect
    collect_steps_per_iteration=1,
    # Params for train
    optimizer=None,
    entropy_regularization=0.0,
    num_epochs=25,
    eval_interval=1,
    total_env_steps=100000,
    replay_buffer_capacity=1001,
    environment_batch_size=1,
    gradient_clipping=None,
    # Params for checkpoints
    policy_checkpoint_interval=5000,
    # Params for summaries and logging
    num_eval_episodes=1,
    log_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    env_args=None,
    seed=0,
):
  """A simple train and eval for PPO."""
  tf.random.set_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  strategy = strategy_utils.get_strategy(tpu=False,
                                         use_gpu=False)  # high cpu machine
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  summary_dir = os.path.join(root_dir, 'summaries')

  collect_env = create_env(
      env_name=env_name,
      env_args=env_args,
  )
  eval_env_args = env_args.copy()
  eval_env = create_env(
      env_name=env_name,
      env_args=eval_env_args,
  )
  logging.info('Successfully created environments')

  observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(collect_env))

  with strategy.scope():
    with tf.name_scope('Actor'):
      actor_net = actor_distribution_network.ActorDistributionNetwork(
          input_tensor_spec=observation_spec,
          output_tensor_spec=action_spec,
          fc_layer_params=[512, 256],
      )
  logging.info('Successfully created actor network')

  with strategy.scope():
    with tf.name_scope('Value'):
      value_net = networks.value_network.ValueNetwork(
          input_tensor_spec=observation_spec,
          fc_layer_params=[512, 256],
      )
  logging.info('Successfully created value network')
  with strategy.scope():
    global_step = train_utils.create_train_step()

    with tf.name_scope('PPOAgent'):
      tf_agent = ppo_clip_agent.PPOClipAgent(
          time_step_spec=time_step_spec,
          action_spec=action_spec,
          optimizer=optimizer,
          actor_net=actor_net,
          value_net=value_net,
          gradient_clipping=gradient_clipping,
          entropy_regularization=entropy_regularization,
          greedy_eval=False,
          importance_ratio_clipping=0.2,
          normalize_observations=False,
          normalize_rewards=False,
          use_gae=True,
          debug_summaries=debug_summaries,
          summarize_grads_and_vars=summarize_grads_and_vars,
          train_step_counter=global_step,
      )
    logging.info('Successfully created PPO agent')

    tf_agent.initialize()
    logging.info('Successfully initialized PPO agent')

  environment_steps_metric = tf_metrics.EnvironmentSteps()
  step_metrics = [
      tf_metrics.NumberOfEpisodes(),
      environment_steps_metric,
  ]
  train_metrics = step_metrics + [
      tf_metrics.AverageReturnMetric(
          batch_size=environment_batch_size, buffer_size=num_eval_episodes
      ),
      tf_metrics.AverageEpisodeLengthMetric(
          batch_size=environment_batch_size, buffer_size=num_eval_episodes
      ),
  ]

  tf_eval_policy = tf_agent.policy
  py_eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_eval_policy, use_tf_function=True)

  tf_collect_policy = tf_agent.collect_policy
  py_collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_collect_policy, use_tf_function=True)

  # Make the replay buffer. Cleared after every iteration
  table_name = 'uniform_table'
  table = reverb.Table(
      table_name,
      max_size=replay_buffer_capacity,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      rate_limiter=reverb.rate_limiters.MinSize(1))

  reverb_server = reverb.Server([table])
  logging.info('Successfully created reverb server')

  reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
      tf_agent.collect_data_spec,
      sequence_length=2,
      table_name=table_name,
      local_server=reverb_server)
  logging.info('Successfully created reverb replay buffer')

  # Create a dataset that samples an amount of data equal to the capacity of the replay buffer
  dataset = reverb_replay.as_dataset(
      sample_batch_size=replay_buffer_capacity, num_steps=2
  ).prefetch(tf.data.experimental.AUTOTUNE)

  experience_dataset_fn = lambda: dataset
  train_checkpointer = common.Checkpointer(
      ckpt_dir=train_dir,
      agent=tf_agent,
      global_step=global_step,
      max_to_keep=1,
      metrics=metric_utils.MetricsGroup(train_metrics, 'train'),
  )
  saved_model_dir = os.path.join(root_dir, 'policies')
  train_checkpointer.initialize_or_restore()

  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      reverb_replay.py_client,
      table_name,
      sequence_length=2,
      stride_length=1)

  collect_actor = actor.Actor(
      env=collect_env,
      policy=py_collect_policy,
      train_step=global_step,
      steps_per_run=collect_steps_per_iteration,
      metrics=actor.collect_metrics(10),
      summary_dir=os.path.join(summary_dir, 'train'),
      observers=[rb_observer]
  )

  eval_actor = actor.Actor(
      env=eval_env,
      policy=py_eval_policy,
      train_step=global_step,
      episodes_per_run=num_eval_episodes,
      metrics=actor.eval_metrics(num_eval_episodes),
      summary_dir=os.path.join(summary_dir, 'eval'), )

  def get_eval_metrics():
    eval_actor.run()
    results = {}
    for metric in eval_actor.metrics:
      results[metric.name] = metric.result()
    return results

  def log_eval_metrics(step, metrics):
    eval_results = (', ').join(
        '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
    print('step = {0}: {1}'.format(step, eval_results))

  # Triggers to save the agent's policy checkpoints.
  learning_triggers = [
      triggers.PolicySavedModelTrigger(
          saved_model_dir=saved_model_dir,
          agent=tf_agent,
          train_step=global_step,
          interval=policy_checkpoint_interval),
      triggers.StepPerSecondLogTrigger(global_step, interval=log_interval),
  ]
  agent_learner = learner.Learner(root_dir=root_dir,
                                  train_step=global_step,
                                  agent=tf_agent,
                                  strategy=strategy,
                                  experience_dataset_fn=experience_dataset_fn,
                                  triggers=learning_triggers,
                                  summary_interval=log_interval,
                                  summary_root_dir=os.path.join(
                                      summary_dir, 'train'),
                                  )

  # Evaluate the agent's policy once before training.
  avg_return = get_eval_metrics()["AverageReturn"]
  returns = [avg_return]

  while environment_steps_metric.result() < total_env_steps:
    # Training.
    collect_actor.run()
    loss_info = agent_learner.run(iterations=num_epochs)

    # Clear the replay buffer since we're learning on-policy
    reverb_replay.clear()

    # Evaluating.
    step = agent_learner.train_step_numpy

    if eval_interval and step % eval_interval == 0:
      metrics = get_eval_metrics()
      log_eval_metrics(step, metrics)
      returns.append(metrics["AverageReturn"])

    if log_interval and step % log_interval == 0:
      print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

  rb_observer.close()
  reverb_server.stop()


def train_mp(_):
  seed = int(os.environ.get('SEED', None))
  root_dir = os.environ.get('ROOT_DIR', None)
  num_epochs = int(os.environ.get('NUM_EPOCHS', None))
  env_batch_size = int(os.environ.get('ENV_BATCH_SIZE', None))
  total_env_steps = int(os.environ.get('TOTAL_ENV_STEPS', None))
  difficulty_level = int(os.environ.get('DIFFICULTY_LEVEL', None))
  eval_interval = int(os.environ.get('EVAL_INTERVAL', None))
  entropy_regularization = float(os.environ.get('ENTROPY_REGULARIZATION', None))
  train_checkpoint_interval = int(
      os.environ.get('INT_SAVE_FREQ', None))
  policy_checkpoint_interval = int(
      os.environ.get('INT_SAVE_FREQ', None))
  log_interval = int(os.environ.get('LOG_INTERVAL', None))
  learning_rate = float(os.environ.get('LEARNING_RATE', None))
  timesteps_per_actorbatch = int(
      os.environ.get('TIMESTEPS_PER_ACTORBATCH', None))
  # motion_file_path = os.environ.get('MOTION_FILE_PATH', None)
  # the motion file dog_pace is in the a2perf package
  motion_file_path = os.path.join(
      os.path.dirname(quadruped_locomotion.__file__), 'motion_imitation',
      'data', 'motions', 'dog_pace.txt')

  # Print extracted and computed values
  print(f'seed: {seed}')
  print(f'root_dir: {root_dir}')
  print(f'env_batch_size: {env_batch_size}')
  print(f'total_env_steps: {total_env_steps}')
  print(f'difficulty_level: {difficulty_level}')
  print(f'eval_interval: {eval_interval}')
  print(f'train_checkpoint_interval: {train_checkpoint_interval}')
  print(f'policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'log_interval: {log_interval}')
  print(f'learning_rate: {learning_rate}')
  print(f'timesteps_per_actorbatch: {timesteps_per_actorbatch}')
  print(f'motion_file_path: {motion_file_path}')
  print(f'num_epochs: {num_epochs}')

  train_eval(
      env_name='QuadrupedLocomotion-v0',
      collect_steps_per_iteration=timesteps_per_actorbatch,
      debug_summaries=False,
      environment_batch_size=env_batch_size,
      eval_interval=eval_interval,
      log_interval=log_interval,
      num_eval_episodes=10,
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      policy_checkpoint_interval=policy_checkpoint_interval,
      num_epochs=num_epochs,
      root_dir=root_dir,
      seed=seed,
      summarize_grads_and_vars=False,
      total_env_steps=total_env_steps,
      entropy_regularization=entropy_regularization,
      replay_buffer_capacity=timesteps_per_actorbatch,
      env_args=dict(
          motion_files=[motion_file_path]
      ),
  )


def train():
  tf_agents.system.multiprocessing.handle_main(train_mp)


if __name__ == '__main__':
  tf_agents.system.multiprocessing.handle_main(
      functools.partial(app.run, train_mp)
  )
