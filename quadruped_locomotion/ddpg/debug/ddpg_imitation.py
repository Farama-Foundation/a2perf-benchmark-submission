import copy
import os
import pickle
import time
from collections import deque

import numpy as np
import tensorflow as tf
from absl import logging
from mpi4py import MPI
from stable_baselines import logger
from stable_baselines.common import tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.math_util import unscale_action, scale_action
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.ddpg import DDPG


def total_eval_episode_reward_logger(rew_acc, rewards, masks, writer, steps):
    """
    calculates the cumulated episode reward, and prints to tensorflow log the output

    :param rew_acc: (np.array float) the total running reward
    :param rewards: (np.array float) the rewards
    :param masks: (np.array bool) the end of episodes
    :param writer: (TensorFlow Session.writer) the writer to log to
    :param steps: (int) the current timestep
    :return: (np.array float) the updated total running reward
    :return: (np.array float) the updated total running reward
    """
    with tf.compat.v1.variable_scope("environment_info", reuse=True):
        for env_idx in range(rewards.shape[0]):
            dones_idx = np.sort(np.argwhere(masks[env_idx]))

            if len(dones_idx) == 0:
                rew_acc[env_idx] += sum(rewards[env_idx])
            else:
                rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
                summary = tf.compat.v1.summary.Summary(value=[tf.compat.v1.Summary.Value(tag="eval_episode_reward", simple_value=rew_acc[env_idx])])
                writer.add_summary(summary, steps + dones_idx[0, 0])
                for k in range(1, len(dones_idx[:, 0])):
                    rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k - 1, 0]:dones_idx[k, 0]])
                    summary = tf.compat.v1.summary.Summary(
                        value=[tf.compat.v1.Summary.Value(tag="eval_episode_reward", simple_value=rew_acc[env_idx])])
                    writer.add_summary(summary, steps + dones_idx[k, 0])
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])

    return rew_acc


class DDPGImitation(DDPG):
    """
    Deep Deterministic Policy Gradient (DDPG) model

    DDPG: https://arxiv.org/pdf/1509.02971.pdf

    :param policy: (DDPGPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param memory_policy: (ReplayBuffer) the replay buffer
        (if None, default to baselines.deepq.replay_buffer.ReplayBuffer)

        .. deprecated:: 2.6.0
            This parameter will be removed in a future version

    :param eval_env: (Gym Environment) the evaluation environment (can be None)
    :param nb_train_steps: (int) the number of training steps
    :param nb_rollout_steps: (int) the number of rollout steps
    :param nb_eval_steps: (int) the number of evaluation steps
    :param param_noise: (AdaptiveParamNoiseSpec) the parameter noise type (can be None)
    :param action_noise: (ActionNoise) the action noise type (can be None)
    :param param_noise_adaption_interval: (int) apply param noise every N steps
    :param tau: (float) the soft update coefficient (keep old values, between 0 and 1)
    :param normalize_returns: (bool) should the critic output be normalized
    :param enable_popart: (bool) enable pop-art normalization of the critic output
        (https://arxiv.org/pdf/1602.07714.pdf), normalize_returns must be set to True.
    :param normalize_observations: (bool) should the observation be normalized
    :param batch_size: (int) the size of the batch for learning the policy
    :param observation_range: (tuple) the bounding values for the observation
    :param return_range: (tuple) the bounding values for the critic output
    :param critic_l2_reg: (float) l2 regularizer coefficient
    :param actor_lr: (float) the actor learning rate
    :param critic_lr: (float) the critic learning rate
    :param clip_norm: (float) clip the gradients (disabled if None)
    :param reward_scale: (float) the value the reward should be scaled by
    :param render: (bool) enable rendering of the environment
    :param render_eval: (bool) enable rendering of the evaluation environment
    :param memory_limit: (int) the max number of transitions to store, size of the replay buffer

        .. deprecated:: 2.6.0
            Use `buffer_size` instead.

    :param buffer_size: (int) the max number of transitions to store, size of the replay buffer
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for DDPG normally but can help exploring when using HER + DDPG.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50,
                 nb_rollout_steps=-1, nb_eval_steps=100, param_noise=None, action_noise=None,
                 normalize_observations=False, tau=0.001, batch_size=128, param_noise_adaption_interval=50,
                 normalize_returns=False, enable_popart=False, observation_range=(-5., 5.), critic_l2_reg=0.,
                 return_range=(-np.inf, np.inf), actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
                 render=False, render_eval=False, memory_limit=None, buffer_size=50000, random_exploration=0.0,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, nb_eval_episodes=1, n_cpu_tf_sess=1):

        super(DDPGImitation, self).__init__(
            policy=policy,
            env=env,
            gamma=gamma,
            memory_policy=memory_policy,
            eval_env=eval_env,
            nb_train_steps=nb_train_steps,
            nb_rollout_steps=nb_rollout_steps,
            nb_eval_steps=nb_eval_steps,
            param_noise=param_noise,
            action_noise=action_noise,
            normalize_observations=normalize_observations,
            tau=tau,
            batch_size=batch_size,
            param_noise_adaption_interval=param_noise_adaption_interval,
            normalize_returns=normalize_returns,
            enable_popart=enable_popart,
            observation_range=observation_range,
            critic_l2_reg=critic_l2_reg,
            return_range=return_range,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            clip_norm=clip_norm,
            reward_scale=reward_scale,
            render=render,
            render_eval=render_eval,
            memory_limit=memory_limit,
            buffer_size=buffer_size,
            random_exploration=random_exploration,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            _init_setup_model=_init_setup_model,
            policy_kwargs=policy_kwargs,
            full_tensorboard_log=full_tensorboard_log,
            seed=seed,
            n_cpu_tf_sess=n_cpu_tf_sess
        )
        self.nb_eval_episodes = nb_eval_episodes
        self.eval_episode_reward = None
        # Parameters.

    def _setup_learn(self):
        """
        Check the environment.
        """
        if self.env is None:
            raise ValueError("Error: cannot train the model without a valid environment, please set an environment with"
                             "set_env(self, env) method.")
        if self.episode_reward is None:
            self.episode_reward = np.zeros((self.n_envs,))
        if self.eval_episode_reward is None:
            self.eval_episode_reward = np.zeros((self.n_envs,))
        if self.ep_info_buf is None:
            self.ep_info_buf = deque(maxlen=100)

    def learn(self, total_timesteps, callback=None, log_interval=None, tb_log_name="DDPG", save_iters=None,
              reset_num_timesteps=True, replay_wrapper=None, save_path=None, eval_iters=1000):
        is_root = MPI.COMM_WORLD.Get_rank() == 0
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            # a list for tensorboard logging, to prevent logging with the same step number, if it already occured
            self.tb_seen_steps = []

            rank = MPI.COMM_WORLD.Get_rank()
            mpi_size = MPI.COMM_WORLD.Get_size()

            if self.verbose >= 2:
                logger.log('Using agent with the following configuration:')
                logger.log(str(self.__dict__.items()))

            eval_episode_rewards_history = deque(maxlen=100)
            episode_rewards_history = deque(maxlen=100)
            episode_successes = []

            with self.sess.as_default(), self.graph.as_default():
                # Prepare everything.
                self._reset()
                obs, info = self.env.reset()
                # Retrieve unnormalized observation for saving into the buffer
                if self._vec_normalize_env is not None:
                    obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                eval_obs = None
                if self.eval_env is not None:
                    eval_obs, eval_info = self.eval_env.reset()
                episode_reward = 0.
                episode_step = 0
                episodes = 0
                step = 0
                total_steps = 0
                iters_so_far = 0

                start_time = time.time()

                epoch_episode_rewards = []
                epoch_episode_steps = []
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                eval_episode_rewards = []
                eval_qs = []
                epoch_actions = []
                epoch_qs = []
                epoch_episodes = 0
                epoch = 0

                callback.on_training_start(locals(), globals())

                while True:
                    callback.on_rollout_start()
                    # Perform rollouts.
                    timesteps_this_iter = 0
                    for _ in range(self.nb_rollout_steps):

                        if total_steps >= total_timesteps:
                            callback.on_training_end()
                            return self

                        # Predict next action.
                        action, q_value = self._policy(obs, apply_noise=True, compute_q=True)
                        assert action.shape == self.env.action_space.shape

                        # Execute next action.
                        if rank == 0 and self.render:
                            self.env.render()

                        # Randomly sample actions from a uniform distribution
                        # with a probability self.random_exploration (used in HER + DDPG)
                        if np.random.rand() < self.random_exploration:
                            # actions sampled from action space are from range specific to the environment
                            # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                            unscaled_action = self.action_space.sample()
                            action = scale_action(self.action_space, unscaled_action)
                        else:
                            # inferred actions need to be transformed to environment action_space before stepping
                            unscaled_action = unscale_action(self.action_space, action)

                        new_obs, reward, done, info = self.env.step(unscaled_action)

                        self.num_timesteps += 1

                        if callback.on_step() is False:
                            callback.on_training_end()
                            return self

                        step += 1
                        total_steps += 1
                        timesteps_this_iter += 1
                        if rank == 0 and self.render:
                            self.env.render()

                        # Book-keeping.
                        epoch_actions.append(action)
                        epoch_qs.append(q_value)

                        # Store only the unnormalized version
                        if self._vec_normalize_env is not None:
                            new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                            reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                        else:
                            # Avoid changing the original ones
                            obs_, new_obs_, reward_ = obs, new_obs, reward

                        self._store_transition(obs_, action, reward_, new_obs_, done,info)
                        obs = new_obs
                        # Save the unnormalized observation
                        if self._vec_normalize_env is not None:
                            obs_ = new_obs_

                        episode_reward += reward_
                        episode_step += 1

                        if writer is not None:
                            ep_rew = np.array([reward_]).reshape((1, -1))
                            ep_done = np.array([done]).reshape((1, -1))
                            tf_util.total_episode_reward_logger(self.episode_reward, ep_rew, ep_done,
                                                                writer, self.num_timesteps)

                        if done:
                            # Episode done.
                            epoch_episode_rewards.append(episode_reward)
                            episode_rewards_history.append(episode_reward)
                            epoch_episode_steps.append(episode_step)
                            episode_reward = 0.
                            episode_step = 0
                            epoch_episodes += 1
                            episodes += 1

                            maybe_is_success = info.get('is_success')
                            if maybe_is_success is not None:
                                episode_successes.append(float(maybe_is_success))

                            self._reset()
                            if not isinstance(self.env, VecEnv):
                                obs, info = self.env.reset()

                    callback.on_rollout_end()
                    # Train.
                    epoch_actor_losses = []
                    epoch_critic_losses = []
                    epoch_adaptive_distances = []
                    for t_train in range(self.nb_train_steps):
                        # Not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size):
                            break

                        # Adapt param noise, if necessary.
                        if len(self.replay_buffer) >= self.batch_size and \
                                t_train % self.param_noise_adaption_interval == 0:
                            distance = self._adapt_param_noise()
                            epoch_adaptive_distances.append(distance)

                        # weird equation to deal with the fact the nb_train_steps will be different
                        # to nb_rollout_steps
                        step = (int(t_train * (self.nb_rollout_steps / self.nb_train_steps)) +
                                self.num_timesteps - self.nb_rollout_steps)

                        critic_loss, actor_loss = self._train_step(step, writer, log=t_train == 0)
                        epoch_critic_losses.append(critic_loss)
                        epoch_actor_losses.append(actor_loss)
                        self._update_target_net()

                    # Evaluate.
                    eval_episode_rewards = []
                    eval_qs = []

                    if self.eval_env is not None and iters_so_far % eval_iters == 0:
                        logger.log("Evaluating...")
                        logger.log(f'iters_so_far: {iters_so_far}')
                        logger.log(f'eval_iters: {eval_iters}')
                        for _ in range(self.nb_eval_episodes):  # Looping over episodes
                            eval_episode_reward = 0.
                            eval_obs, eval_info = self.eval_env.reset()

                            while True:  # Inner loop for each step of the episode
                                if total_steps >= total_timesteps:
                                    return self

                                eval_action, eval_q = self._policy(eval_obs, apply_noise=False, compute_q=True)
                                unscaled_action = unscale_action(self.action_space, eval_action)
                                eval_obs, eval_r, eval_done, _ = self.eval_env.step(unscaled_action)

                                if self.render_eval:
                                    self.eval_env.render()

                                eval_episode_reward += eval_r

                                eval_qs.append(eval_q)

                                if writer is not None:
                                    ep_rew = np.array([eval_r]).reshape((1, -1))
                                    ep_done = np.array([eval_done]).reshape((1, -1))
                                    total_eval_episode_reward_logger(self.eval_episode_reward, ep_rew, ep_done,
                                                                     writer, self.num_timesteps)

                                if eval_done:
                                    break  # If episode is done, break the inner loop and start next episode

                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)

                    # Log stats.
                    # XXX shouldn't call np.mean on variable length lists
                    duration = time.time() - start_time
                    stats = self._get_stats()
                    combined_stats = stats.copy()
                    combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
                    combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
                    combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
                    combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
                    combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
                    combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
                    combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
                    combined_stats['total/timesteps_this_iter'] = timesteps_this_iter
                    if len(epoch_adaptive_distances) != 0:
                        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
                    combined_stats['total/duration'] = duration
                    combined_stats['total/steps_per_second'] = float(step) / float(duration)
                    combined_stats['total/episodes'] = episodes
                    combined_stats['rollout/episodes'] = epoch_episodes
                    combined_stats['rollout/actions_std'] = np.std(epoch_actions)

                    # Evaluation statistics.
                    if self.eval_env is not None:
                        combined_stats['eval/return'] = np.mean(eval_episode_rewards)
                        combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                        combined_stats['eval/Q'] = np.mean(eval_qs)
                        combined_stats['eval/episodes'] = len(eval_episode_rewards)

                    def as_scalar(scalar):
                        """
                        check and return the input if it is a scalar, otherwise raise ValueError

                        :param scalar: (Any) the object to check
                        :return: (Number) the scalar if x is a scalar
                        """
                        if isinstance(scalar, np.ndarray):
                            assert scalar.size == 1
                            return scalar[0]
                        elif np.isscalar(scalar):
                            return scalar
                        else:
                            raise ValueError('expected scalar, got %s' % scalar)

                    # Define which keys are sum-able and which are average-able
                    sum_keys = ['total/timesteps_this_iter', 'total/episodes', 'rollout/episodes', 'eval/episodes']
                    average_keys = list(set(combined_stats.keys()) - set(sum_keys))

                    # Aggregate stats using MPI allreduce with sorted keys
                    sorted_keys = sorted(combined_stats.keys())
                    all_combined_stats = np.array([as_scalar(combined_stats[k]) for k in sorted_keys])
                    combined_stats_sums = MPI.COMM_WORLD.Allreduce(
                        copy.deepcopy(all_combined_stats), all_combined_stats, op=MPI.SUM)

                    summed_stats = {k: all_combined_stats[i] for i, k in enumerate(sorted_keys) if k in sum_keys}
                    averaged_stats = {k: all_combined_stats[i] / mpi_size for i, k in enumerate(sorted_keys) if
                                      k in average_keys}

                    # Update combined_stats
                    combined_stats.update(summed_stats)
                    combined_stats.update(averaged_stats)

                    self.num_timesteps += combined_stats['total/timesteps_this_iter']
                    total_steps += combined_stats['total/timesteps_this_iter']

                    if rank == 0:
                        # print out the dictionary nicely to show "after allreduce"
                        logger.info('*********** Iteration %i ************' % iters_so_far)
                        for key in sorted_keys:
                            logger.record_tabular(key, combined_stats[key])
                        logger.record_tabular('total/timesteps', self.num_timesteps)

                        if len(episode_successes) > 0:
                            logger.logkv("success rate", np.mean(episode_successes[-100:]))
                        logger.dump_tabular()
                        logger.info('')

                        logdir = logger.get_dir()
                        if logdir:
                            if hasattr(self.env, 'get_state'):
                                with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as file_handler:
                                    pickle.dump(self.env.get_state(), file_handler)
                            if self.eval_env and hasattr(self.eval_env, 'get_state'):
                                with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as file_handler:
                                    pickle.dump(self.eval_env.get_state(), file_handler)

                    iters_so_far += 1

                    if is_root and (save_path is not None) and (iters_so_far % save_iters == 0):
                        logger.info(
                            f'Saving model at {save_path} at {iters_so_far} iterations and {self.num_timesteps} steps')

                        path = os.path.join(save_path, '{}_{}_steps'.format('rl_policy', int(self.num_timesteps)))
                        self.save(path)
