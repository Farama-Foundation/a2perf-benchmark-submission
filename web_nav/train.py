import multiprocessing
import os
import random
import time
import logging
import gin
import gymnasium as gym
import numpy as np
import tensorflow as tf
import tf_agents
import absl
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from rl_perf.domains.web_nav.gwob.CoDE import q_networks
from rl_perf.domains.web_nav.gwob.CoDE import vocabulary_node


class DQNLSTM(network.Network):

    def __init__(
            self,
            observation_spec,
            action_spec,
            state_spec=(),
            name='DQNLSTM',
            **kwargs,
    ):
        super().__init__(
            input_tensor_spec=observation_spec, state_spec=state_spec, name=name
        )
        self._action_spec = action_spec
        self._lstm = q_networks.DQNWebLSTM(**kwargs)

    def call(self, observation, step_type=None, network_state=(), training=False):
        q_values, _ = self._lstm(observation, is_training=training)
        return q_values, network_state


def remove_config_lines(config_string, key):
    lines = config_string.split('\n')
    lines = [line for line in lines if not line.startswith(key)]
    return '\n'.join(lines)


def create_env(env_seed: int, env_name='CartPole-v0', difficulty=None, global_vocab=None, env_args=None):
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
        num_iterations=100000,
        # Params for collect
        initial_collect_steps=5,
        collect_steps_per_iteration=1,
        epsilon_greedy=0.05,
        max_vocab_size=500,
        replay_buffer_capacity=100000,
        # Params for target update
        target_update_tau=0.05,
        target_update_period=5000,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=32,
        environment_batch_size=1,
        learning_rate=1e-4,
        n_step_update=1,
        gamma=0.99,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        use_tf_functions=True,
        # Params for eval
        num_eval_episodes=10,
        eval_interval=1000,
        # Params for checkpoints
        train_checkpoint_interval=10000,
        policy_checkpoint_interval=5000,
        rb_checkpoint_interval=20000,
        # Params for summaries and logging
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
        os.path.join(summary_dir, 'train'), flush_millis=summaries_flush_secs * 1000
    )
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        os.path.join(summary_dir, 'eval'), flush_millis=summaries_flush_secs * 1000
    )
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
    ]

    global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int64)
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    global_vocab = (
        vocabulary_node.LockedVocabulary(
            multiprocessing_lock=lock)
    )
    envs = [lambda: create_env(seed + i, env_name=env_name, difficulty=difficulty, global_vocab=global_vocab,
                               env_args=env_args) for i
            in range(environment_batch_size)]
    eval_env = tf_agents.environments.ParallelPyEnvironment(
        [lambda: create_env(seed + environment_batch_size, env_name=env_name,
                            difficulty=difficulty,
                            global_vocab=global_vocab,
                            env_args=env_args)])
    eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)
    parallel_py_env = tf_agents.environments.ParallelPyEnvironment(envs, blocking=True, start_serially=True)
    tf_env = tf_py_environment.TFPyEnvironment(parallel_py_env)

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    with tf.name_scope('DQNAgent'):
        q_net = DQNLSTM(
            observation_spec=observation_spec,
            action_spec=action_spec,
            state_spec=(),
            vocab_size=max_vocab_size
            if max_vocab_size is not None
            else tf_env.pyenv.envs[0].env.local_vocab.max_vocabulary_size,
            profile_value_dropout=0.0,
            q_min=None,
            q_max=None,
            embedding_dim=100,
            name='q_network',
            latent_dim=50,
            return_state_value=True,
        )
        tf_agent = dqn_agent.DqnAgent(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            q_network=q_net,
            epsilon_greedy=epsilon_greedy,
            n_step_update=n_step_update,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=learning_rate
            ),
            td_errors_loss_fn=common.element_wise_huber_loss,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step,
            name='dqn_agent',
        )
        tf_agent.initialize()

    # Make the replay buffer.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity,
        device='/gpu:*',
        # device='/cpu:*',
    )
    replay_observer = [replay_buffer.add_batch]

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
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec()
    )
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
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer,
    )

    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=initial_collect_steps,
    )

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=collect_steps_per_iteration,
    )

    if use_tf_functions:
        initial_collect_driver.run = common.function(initial_collect_driver.run)
        collect_driver.run = common.function(collect_driver.run)
        tf_agent.train = common.function(tf_agent.train)

    if replay_buffer.num_frames() == 0:
        # Collect initial replay data.
        absl.logging.info(
            'Initializing replay buffer by collecting experience for %d steps '
            'with a random policy.',
            initial_collect_steps,
        )
        initial_collect_driver.run()

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

    dataset = (
        replay_buffer.as_dataset(sample_batch_size=batch_size,
                                 num_steps=2,
                                 num_parallel_calls=tf.data.AUTOTUNE
                                 )
        .unbatch()
        .filter(_filter_invalid_transition)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    # Dataset generates trajectories with shape [Bx2x...]
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
            absl.logging.info('step = %d, loss = %f', global_step_val, train_loss.loss)
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

        if iters_so_far % eval_interval == 0:
            results = metric_utils.eager_compute(
                eval_metrics,
                eval_tf_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix='Metrics',
                use_function=False,
            )
            if eval_metrics_callback is not None:
                eval_metrics_callback(results, global_step)
            metric_utils.log_metrics(eval_metrics)

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
    # Extract environment variables
    seed = int(os.environ.get('SEED', None))
    root_dir = os.environ.get('ROOT_DIR', None)
    env_batch_size = int(os.environ.get('ENV_BATCH_SIZE', None))
    total_env_steps = int(os.environ.get('TOTAL_ENV_STEPS', None))
    difficulty_level = int(os.environ.get('DIFFICULTY_LEVEL', None))
    eval_interval = int(os.environ.get('EVAL_INTERVAL', None))
    train_checkpoint_interval = int(os.environ.get('TRAIN_CHECKPOINT_INTERVAL', None))
    policy_checkpoint_interval = int(os.environ.get('POLICY_CHECKPOINT_INTERVAL', None))
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
        replay_buffer_capacity=rb_capacity,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        eval_interval=eval_interval,
        collect_steps_per_iteration=timesteps_per_actorbatch,
        train_checkpoint_interval=train_checkpoint_interval,
        policy_checkpoint_interval=policy_checkpoint_interval,
        rb_checkpoint_interval=rb_checkpoint_interval,
        initial_collect_steps=timesteps_per_actorbatch,
        log_interval=log_interval,
        summary_interval=summary_interval,
        env_args=dict()
    )


def train():
    logging.basicConfig(level=logging.WARNING)
    tf_agents.system.multiprocessing.handle_main(train_mp)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    tf_agents.system.multiprocessing.handle_main(train_mp)
