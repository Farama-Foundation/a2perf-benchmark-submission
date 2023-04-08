import os
import random
import time

import gin
import gym
import numpy as np
import pandas as pd
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from absl import logging
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from rl_perf.domains.web_nav.CoDE import q_networks


class DQNLSTM(network.Network):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 state_spec=(),
                 name='DQNLSTM', **kwargs):
        super().__init__(input_tensor_spec=observation_spec, state_spec=state_spec, name=name)
        self._action_spec = action_spec
        self._lstm = q_networks.DQNWebLSTM(**kwargs)

    def call(self, observation, step_type=None, network_state=(), training=False):
        q_values, _ = self._lstm(observation, is_training=training)
        return q_values, network_state


@gin.configurable
def train_eval(
        root_dir,
        env_name='CartPole-v0',
        use_gpu=False,
        num_iterations=100000,
        train_sequence_length=1,
        # Params for collect
        initial_collect_steps=20,
        collect_steps_per_iteration=1,
        epsilon_greedy=0.1,
        max_vocab_size=500,
        replay_buffer_capacity=100000,
        # Params for target update
        target_update_tau=0.05,
        target_update_period=5,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=64,
        learning_rate=1e-3,
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
        seed=0, ):
    """A simple train and eval for DQN."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
            train_dir, flush_millis=summaries_flush_secs * 1000)

    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
            eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
            ]

    os.makedirs(eval_dir, exist_ok=True)
    with  open(os.path.join(eval_dir, 'eval_summary.csv'), 'w') as eval_file:
        eval_file.write(','.join([metric.name for metric in eval_metrics]))

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):
        tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(environment_name=env_name,
                                                                  spec_dtype_map={gym.spaces.Discrete: np.int32},
                                                                  gym_kwargs={'difficulty': 1, 'seed': seed}))
        eval_tf_env = tf_py_environment.TFPyEnvironment(
                suite_gym.load(env_name, gym_kwargs={'difficulty': 1, 'seed': seed + 10}))

        if train_sequence_length != 1 and n_step_update != 1:
            raise NotImplementedError(
                    'train_eval does not currently support n-step updates with stateful '
                    'networks (i.e., RNNs)')

        action_spec = tf_env.action_spec()
        num_actions = action_spec.maximum - action_spec.minimum + 1

        with tf.name_scope('DQNAgent'):

            q_net = DQNLSTM(
                    observation_spec=tf_env.observation_spec(),
                    action_spec=tf_env.action_spec(),
                    state_spec=(),
                    vocab_size=max_vocab_size if max_vocab_size is not None else tf_env.pyenv.envs[
                        0].env.local_vocab.max_vocabulary_size,
                    profile_value_dropout=0.0,
                    q_min=None,
                    q_max=None,
                    embedding_dim=100,
                    name='q_network',
                    latent_dim=50,
                    return_state_value=True)

            tf_agent = dqn_agent.DqnAgent(
                    tf_env.time_step_spec(),
                    tf_env.action_spec(),
                    q_network=q_net,
                    epsilon_greedy=epsilon_greedy,
                    n_step_update=n_step_update,
                    target_update_tau=target_update_tau,
                    target_update_period=target_update_period,
                    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
                    td_errors_loss_fn=common.element_wise_huber_loss,
                    gamma=gamma,
                    reward_scale_factor=reward_scale_factor,
                    gradient_clipping=gradient_clipping,
                    debug_summaries=debug_summaries,
                    summarize_grads_and_vars=summarize_grads_and_vars,
                    train_step_counter=global_step,
                    name='dqn_agent'
                    )
            tf_agent.initialize()

        train_metrics = [
                tf_metrics.NumberOfEpisodes(),
                tf_metrics.EnvironmentSteps(),
                tf_metrics.AverageReturnMetric(),
                tf_metrics.AverageEpisodeLengthMetric(),
                ]
        os.makedirs(train_dir, exist_ok=True)
        with open(os.path.join(train_dir, 'train_summary.csv'), 'w') as train_file:
            train_file.write(','.join([metric.name for metric in train_metrics]))

        eval_policy = tf_agent.policy
        collect_policy = tf_agent.collect_policy

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=tf_agent.collect_data_spec,
                batch_size=tf_env.batch_size,
                max_length=replay_buffer_capacity,
                device='gpu:0' if use_gpu else 'cpu:0'
                )
        initial_collect_policy = random_tf_policy.RandomTFPolicy(
                tf_env.time_step_spec(), tf_env.action_spec(), validate_args=True)

        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
                tf_env,
                initial_collect_policy,
                observers=[replay_buffer.add_batch] + train_metrics,
                num_steps=initial_collect_steps)
        collect_driver = dynamic_step_driver.DynamicStepDriver(
                tf_env,
                collect_policy,
                observers=[replay_buffer.add_batch] + train_metrics,
                num_steps=collect_steps_per_iteration)

        train_checkpointer = common.Checkpointer(
                ckpt_dir=train_dir,
                agent=tf_agent,
                global_step=global_step,
                max_to_keep=3,
                metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
        policy_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(train_dir, 'policy'),
                max_to_keep=3,
                policy=eval_policy,
                global_step=global_step)
        rb_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
                max_to_keep=3,
                replay_buffer=replay_buffer)

        train_checkpointer.initialize_or_restore()
        rb_checkpointer.initialize_or_restore()

        if use_tf_functions:
            initial_collect_driver.run = common.function(initial_collect_driver.run)
            collect_driver.run = common.function(collect_driver.run)
            tf_agent.train = common.function(tf_agent.train)

        # Collect initial replay data.
        logging.info(
                'Initializing replay buffer by collecting experience for %d steps with '
                'a random policy.', initial_collect_steps)
        dynamic_step_driver.DynamicStepDriver(
                tf_env,
                initial_collect_policy,
                observers=[replay_buffer.add_batch] + train_metrics,
                num_steps=initial_collect_steps).run()

        results = metric_utils.eager_compute(
                eval_metrics,
                eval_tf_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix='Metrics',
                )
        if eval_metrics_callback is not None:
            eval_metrics_callback(results, global_step.numpy())
        metric_utils.log_metrics(eval_metrics)

        # Save initial eval metrics
        results = {k: v.numpy() for k, v in results.items()}
        eval_df = pd.read_csv(os.path.join(eval_dir, 'eval_summary.csv'))
        eval_df = eval_df.append(results, ignore_index=True)
        eval_df.to_csv(os.path.join(eval_dir, 'eval_summary.csv'), index=False)
        del eval_df

        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)

        timed_at_step = global_step.numpy()
        time_acc = 0

        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=batch_size,
                num_steps=train_sequence_length + 1).prefetch(3)
        iterator = iter(dataset)

        def train_step():
            experience, _ = next(iterator)
            return tf_agent.train(experience)

        if use_tf_functions:
            train_step = common.function(train_step)

        for _ in range(num_iterations):
            start_time = time.time()
            global_step_val = global_step.numpy()
            time_step, policy_state = collect_driver.run(
                    time_step=time_step,
                    policy_state=policy_state,
                    )
            for _ in range(train_steps_per_iteration):
                train_loss = train_step()
            time_acc += time.time() - start_time

            if global_step.numpy() % log_interval == 0:
                logging.info('step = %d, loss = %f', global_step.numpy(),
                             train_loss.loss)
                steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
                logging.info('%.3f steps/sec', steps_per_sec)
                tf.compat.v2.summary.scalar(
                        name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                timed_at_step = global_step.numpy()
                time_acc = 0

            if global_step_val % train_checkpoint_interval == 0:
                train_checkpointer.save(global_step=global_step_val)

            if global_step_val % policy_checkpoint_interval == 0:
                policy_checkpointer.save(global_step=global_step_val)

            if global_step_val % rb_checkpoint_interval == 0:
                rb_checkpointer.save(global_step=global_step_val)

            if global_step.numpy() % summary_interval == 0:
                csv_results = []
                for train_metric in train_metrics:
                    metric_val = train_metric.result()
                    csv_results.append(metric_val.numpy())
                    train_metric.tf_summaries(train_step=global_step, step_metrics=train_metrics[:2])

                train_df = pd.read_csv(os.path.join(train_dir, 'train_summary.csv'))
                train_df = train_df.append(pd.Series(csv_results, index=train_df.columns), ignore_index=True)
                train_df.to_csv(os.path.join(train_dir, 'train_summary.csv'), index=False)
                del train_df

            if global_step.numpy() % eval_interval == 0:
                results = metric_utils.eager_compute(
                        eval_metrics,
                        eval_tf_env,
                        eval_policy,
                        num_episodes=num_eval_episodes,
                        train_step=global_step,
                        summary_writer=eval_summary_writer,
                        summary_prefix='Metrics',
                        )
                if eval_metrics_callback is not None:
                    eval_metrics_callback(results, global_step.numpy())
                metric_utils.log_metrics(eval_metrics)
                results = {k: v.numpy() for k, v in results.items()}
                eval_df = pd.read_csv(os.path.join(eval_dir, 'eval_summary.csv'))
                # add row to df based on results dictionary
                eval_df = eval_df.append(results, ignore_index=True)
                eval_df.to_csv(os.path.join(eval_dir, 'eval_summary.csv'), index=False)
                del eval_df

        return train_loss


def train():
    gin.parse_config_file('../rlperf_benchmark_submission/train.gin')
    train_eval()


if __name__ == '__main__':
    train()
