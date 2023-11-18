import os

import gymnasium
import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from train import DQNLSTM

from rl_perf.domains.web_nav.CoDE import vocabulary_node


def load_model():
    # get root dir from env var

    root_dir = os.environ['ROOT_DIR']
    # seed = os.environ['SEED']
    # env_batch_size = os.environ['ENV_BATCH_SIZE']

    root_dir = root_dir
    print(root_dir)

    env_name = 'WebNavigation-v0'
    learning_rate = 1e-4
    max_vocab_size = 500
    seed = 32
    train_dir = os.path.join(root_dir, 'train')

    # Load the global vocabulary
    global_vocab_dict = np.load(os.path.join(train_dir, 'global_vocab.npy'), allow_pickle=True).item()
    global_vocab = vocabulary_node.LockedVocabulary()
    global_vocab.restore(dict(global_vocab=global_vocab_dict))
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(environment_name=env_name,
                                                              spec_dtype_map={gym.spaces.Discrete: np.int32},
                                                              gym_kwargs={'designs': designs, 'seed': seed,
                                                                          'global_vocabulary': global_vocab, }))

    global_step = tf.compat.v1.train.get_or_create_global_step()
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
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
        td_errors_loss_fn=common.element_wise_huber_loss,
        name='dqn_agent'
    )
    eval_policy = tf_agent.policy

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        max_to_keep=3,
        policy=eval_policy,
        global_step=global_step)
    policy_checkpointer.initialize_or_restore()

    return eval_policy


def preprocess_observation(observation):
    # Write your code here to preprocess the observation. This function should return the preprocessed observation.
    time_step = ts.TimeStep(step_type=ts.StepType.FIRST, reward=0.0, discount=1.0, observation=observation)

    # Convert the single timestep into a batch of size 1
    time_step = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), time_step)

    return time_step


def infer_once(model, observation):
    # Write your code here to run inference on the model. This function should return the output of the model.

    observation = preprocess_observation(observation)

    action_step = model.action(time_step=observation)
    action = tf.nest.map_structure(lambda t: tf.squeeze(t, axis=0), action_step.action)
    return action
