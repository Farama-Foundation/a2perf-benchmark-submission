import multiprocessing
import os

os.environ['WRAPT_DISABLE_EXTENSIONS'] = '1'
import random
import time
import functools
import absl
import gin
from absl import app
from absl import flags
from absl import logging
import gymnasium as gym
import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import network
from tf_agents.policies import greedy_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.networks import actor_distribution_network
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


class PPOLSTMActor(actor_distribution_network.ActorDistributionNetwork):
  """Creates an actor network."""

  def __init__(self, observation_spec, action_spec, state_spec=(),
      name='PPOLSTMActor', **kwargs):
    # Initialize the parent class
    self.lstm = networks.WebLSTMActor(**kwargs)
    super(PPOLSTMActor, self).__init__(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        preprocessing_layers=None,

        preprocessing_combiner=self.lstm,
        fc_layer_params=None,
        dropout_layer_params=None,
        activation_fn=tf.keras.activations.relu,
        kernel_initializer=None,
        name=name)


class PPOLSTMCritic(network.Network):
  """Creates a critic network."""

  def __init__(
      self,
      observation_spec,
      action_spec,
      state_spec=(),
      name='PPOLSMTCritic',
      **kwargs,
  ):
    super().__init__(
        input_tensor_spec=observation_spec, state_spec=state_spec, name=name
    )
    self._action_spec = action_spec
    self.lstm = networks.WebLSTMCritic(**kwargs)

  def call(self, observation, step_type=None, network_state=(), training=False):
    value_estimate = self.lstm(observation, is_training=training)
    return value_estimate, network_state


@gin.configurable
def train_eval(
    root_dir,
    env_name='WebNavigation-v0',
    difficulty=None,
    num_iterations=100000,
    # Params for collect
    collect_steps_per_iteration=1,
    max_vocab_size=500,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=32,
    replay_buffer_capacity=1001,
    environment_batch_size=1,
    learning_rate=1e-4,
    gamma=0.99,
    reward_scale_factor=1.0,
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
    eval_metrics_callback=None,
    env_args=None,
    seed=0,
):
  """A simple train and eval for DQN."""

  print('IOU: Starting the training script. Print')
  tf.random.set_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')

  summary_dir = os.path.join(root_dir, 'summaries')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      os.path.join(summary_dir, 'train'),
      flush_millis=summaries_flush_secs * 1000
  )
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      os.path.join(summary_dir, 'eval'),
      flush_millis=summaries_flush_secs * 1000
  )
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]

  global_step = tf.Variable(0, trainable=False, name="global_step",
                            dtype=tf.int64)
  manager = multiprocessing.Manager()
  lock = manager.Lock()
  global_vocab = (
      vocabulary_node.LockedVocabulary(max_vocabulary_size=max_vocab_size,
                                       multiprocessing_lock=lock)
  )
  envs = [lambda: create_env(seed + i, env_name=env_name, difficulty=difficulty,
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

  time_step_spec = tf_env.time_step_spec()
  observation_spec = time_step_spec.observation
  action_spec = tf_env.action_spec()

  with tf.name_scope('PPOAgent'):

    actor_net = PPOLSTMActor(
        observation_spec=observation_spec,
        action_spec=action_spec,
        state_spec=(),
        vocab_size=max_vocab_size
        if max_vocab_size is not None
        else tf_env.pyenv.envs[0].env.local_vocab.max_vocabulary_size,
        profile_value_dropout=0.0,
        embedding_dim=100,
        name='actor',
        latent_dim=50,
    )
    extra_args = dict(fw_bs_encoder=actor_net.lstm._fw_bs_encoder if hasattr(
        actor_net.lstm, '_fw_bs_encoder') else None,
                      dom_encoder_bw=actor_net.lstm._dom_encoder_bw if hasattr(
                          actor_net.lstm, '_dom_encoder_bw') else None)
    extra_args = {k: v for k, v in extra_args.items() if v is not None}

    critic_net = PPOLSTMCritic(
        observation_spec=observation_spec,
        action_spec=action_spec,
        embedder=actor_net.lstm._embedder,
        dom_element_encoder=actor_net.lstm._dom_element_encoder,
        dom_encoder=actor_net.lstm._dom_encoder,
        profile_encoder=actor_net.lstm._profile_encoder,
        vocab_size=max_vocab_size
        if max_vocab_size is not None
        else tf_env.pyenv.envs[0].env.local_vocab.max_vocabulary_size,
        profile_value_dropout=0.0,
        embedding_dim=100,
        name='actor',
        latent_dim=50,
        **extra_args
    )

    tf_agent = ppo_agent.PPOAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, ),
        actor_net=actor_net,  # Optional, default is None
        value_net=critic_net,  # Optional, default is None
        greedy_eval=True,  # Optional, default is True
        importance_ratio_clipping=0.0,  # Optional, default is 0.0
        lambda_value=0.95,  # Optional, default is 0.95
        discount_factor=gamma,  # Optional, default is 0.99
        entropy_regularization=0.0,  # Optional, default is 0.0
        policy_l2_reg=0.0,  # Optional, default is 0.0
        value_function_l2_reg=0.0,  # Optional, default is 0.0
        shared_vars_l2_reg=0.0,  # Optional, default is 0.0
        value_pred_loss_coef=0.5,  # Optional, default is 0.5
        num_epochs=1,  # Optional, default is 25
        use_gae=False,  # Optional, default is False
        use_td_lambda_return=False,  # Optional, default is False
        normalize_rewards=True,  # Optional, default is True
        reward_norm_clipping=10.0,  # Optional, default is 10.0
        normalize_observations=True,  # Optional, default is True
        log_prob_clipping=0.0,  # Optional, default is 0.0
        kl_cutoff_factor=2.0,  # Optional, default is 2.0
        kl_cutoff_coef=1000.0,  # Optional, default is 1000.0
        initial_adaptive_kl_beta=1.0,  # Optional, default is 1.0
        adaptive_kl_target=0.01,  # Optional, default is 0.01
        adaptive_kl_tolerance=0.3,  # Optional, default is 0.3
        gradient_clipping=gradient_clipping,  # Optional, default is None
        value_clipping=None,  # Optional, default is None
        check_numerics=False,  # Optional, default is False
        compute_value_and_advantage_in_train=True,  # Optional, default is True
        update_normalizers_in_train=True,  # Optional, default is True
        aggregate_losses_across_replicas=True,  # Optional, default is True
        debug_summaries=debug_summaries,  # Optional, default is False
        summarize_grads_and_vars=summarize_grads_and_vars,
        # Optional, default is False
        train_step_counter=global_step,  # Optional, default is None
        name='PPOAgent',  # Optional, default is 'PPOAgent'
    )

    tf_agent.initialize()

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(
            buffer_size=num_eval_episodes, batch_size=tf_env.batch_size
        ),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=num_eval_episodes, batch_size=tf_env.batch_size
        ),
    ]

    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    collect_policy = tf_agent.collect_policy

    train_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'train'),
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
    )
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step,
    )

    train_checkpointer.initialize_or_restore()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=environment_batch_size,
        max_length=replay_buffer_capacity,
    )

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=collect_steps_per_iteration,
    )

    if use_tf_functions:
      collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)

    # Compute eval metrics once at the beginning of training
    results = metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
    )

    # Compute train metrics once at the beginning of training
    with train_summary_writer.as_default():
      for train_metric in train_metrics:
        metric_value = train_metric.result()
        # Prefix the metric's name with 'Metrics/'
        metric_name = f"Metrics/{train_metric.name}"
        tf.summary.scalar(metric_name, metric_value, step=global_step)
    train_summary_writer.flush()

    if eval_metrics_callback is not None:
      eval_metrics_callback(results, global_step.numpy())
    metric_utils.log_metrics(eval_metrics)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0

    # Prepare replay buffer as dataset with invalid transitions filtered.
    def _filter_invalid_transition(trajectories, _):
      return ~trajectories.is_boundary()[0]

    # We'll use replay_buffer.as_dataset but we need it to give us ALL the data
    # since PPO needs to do multiple passes over the data.

    # dataset = replay_buffer.as_dataset(
    #     num_parallel_calls=tf.data.AUTOTUNE,
    #     # sample_batch_size=1,
    #     # num_steps=2,
    #     single_deterministic_pass=False,
    # ).unbatch().filter(_filter_invalid_transition).batch(batch_size).prefetch(
    #     tf.data.AUTOTUNE
    # )
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=tf.data.AUTOTUNE,
        sample_batch_size=collect_steps_per_iteration * environment_batch_size,
        num_steps=1,
    ).batch(batch_size=32).prefetch(tf.data.AUTOTUNE)
    iterator = iter(dataset)

  def train_step():
    experience, _ = next(iterator)
    return tf_agent.train(experience)

  if use_tf_functions:
    train_step = common.function(train_step)

  iters_so_far = 0
  while iters_so_far < num_iterations:

    # create tf summaryh for iters_so_far
    with train_summary_writer.as_default():
      tf.summary.scalar('iters_so_far', iters_so_far, step=iters_so_far)
    start_time = time.time()
    time_step, policy_state = collect_driver.run(
        time_step=time_step,
        policy_state=policy_state,
    )

    for _ in range(train_steps_per_iteration):
      train_loss = train_step()

    time_acc += time.time() - start_time

    global_step_val = global_step.numpy()

    if iters_so_far % log_interval == 0:
      absl.logging.info('step = %d, loss = %f', global_step_val,
                        train_loss.loss)
      print('step = %d, loss = %f', global_step_val, train_loss.loss)
      steps_per_sec = (global_step_val - timed_at_step) / time_acc
      absl.logging.info('%.3f steps/sec', steps_per_sec)
      print('%.3f steps/sec', steps_per_sec)
      tf.compat.v2.summary.scalar(
          name='global_steps_per_sec', data=steps_per_sec, step=iters_so_far
      )
      timed_at_step = global_step_val
      time_acc = 0

    if iters_so_far % summary_interval == 0:
      with train_summary_writer.as_default():
        for train_metric in train_metrics:
          metric_value = train_metric.result()
          # Prefix the metric's name with 'Metrics/'
          metric_name = f"Metrics/{train_metric.name}"
          tf.summary.scalar(metric_name, metric_value, step=global_step)
      train_summary_writer.flush()

    if iters_so_far % train_checkpoint_interval == 0:
      train_checkpointer.save(global_step=global_step)
      tokens = global_vocab.save()
      np.save(os.path.join(train_dir, f'vocab_{global_step_val}'), tokens)

    if iters_so_far % policy_checkpoint_interval == 0:
      policy_checkpointer.save(global_step=global_step)

    iters_so_far += 1

  manager.shutdown()
  tf_env.close()
  eval_tf_env.close()
  return train_loss


def train_mp(_):
  logging.set_verbosity(logging.INFO)
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
  batch_size = int(os.environ.get('BATCH_SIZE', None))
  summary_interval = int(os.environ.get('SUMMARY_INTERVAL', None))
  timesteps_per_actorbatch_param = int(
      os.environ.get('TIMESTEPS_PER_ACTORBATCH', None)
  )
  batched_total_env_steps = total_env_steps // env_batch_size
  timesteps_per_actorbatch = max(
      1, timesteps_per_actorbatch_param // env_batch_size
  )
  num_iterations = max(1, batched_total_env_steps // timesteps_per_actorbatch)

  # Print extracted and computed values
  print(f'seed: {seed}')
  print(f'root_dir: {root_dir}')
  print(f'env_batch_size: {env_batch_size}')
  print(f'batch_size: {batch_size}')
  print(f'total_env_steps: {total_env_steps}')
  print(f'num_iterations: {num_iterations}')
  print(f'difficulty_level: {difficulty_level}')
  print(f'train_steps_per_iteration: {timesteps_per_actorbatch_param}')
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
      batch_size=batch_size,
      environment_batch_size=env_batch_size,
      train_steps_per_iteration=timesteps_per_actorbatch,
      num_iterations=num_iterations,
      learning_rate=learning_rate,
      collect_steps_per_iteration=timesteps_per_actorbatch,
      train_checkpoint_interval=train_checkpoint_interval,
      policy_checkpoint_interval=policy_checkpoint_interval,
      log_interval=log_interval,
      summary_interval=summary_interval,
      env_args=dict()
  )


def train():
  tf_agents.system.multiprocessing.handle_main(train_mp)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  tf_agents.system.multiprocessing.handle_main(
      functools.partial(app.run, train_mp))
