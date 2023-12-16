from absl import flags

# delete all absl flags
for name in list(flags.FLAGS):
    delattr(flags.FLAGS, name)

import os
import numpy as np
import gymnasium
import tensorflow as tf
from absl import app
from tf_agents.train.utils import spec_utils, train_utils, strategy_utils
from tf_agents.trajectories import time_step as ts

from learning import static_feature_cache
from learning.agent import create_circuit_ppo_agent
from learning.train_ppo import try_load_checkpoint
from model import create_models_lib
import sys
from a2perf.domains import circuit_training

FLAGS = flags.FLAGS


def load_model():
    """This function loads and prepares the model for inference."""
    FLAGS(sys.argv[:1])  # need to explicitly to tell flags library to parse argv before you can access FLAGS.xxx

    root_dir = os.environ['ROOT_DIR']
    seed = os.environ['GLOBAL_SEED']

    env = gym.make('CircuitTraining-v0')
    observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
        spec_utils.get_tensor_specs(env))
    static_features = env.wrapped_env().get_static_obs()
    cache = static_feature_cache.StaticFeatureCache()
    cache.add_static_feature(static_features)

    strategy = strategy_utils.get_strategy(
        strategy_utils.TPU.value, strategy_utils.USE_GPU.value
    )

    with strategy.scope():
        actor_net, value_net = create_models_lib.create_models_fn(
            'generalization',
            observation_tensor_spec,
            action_tensor_spec,
            cache.get_all_static_features(),
            seed=int(seed),
        )

        actor_net.create_variables(training=False)
        value_net.create_variables(training=False)

        init_train_step = try_load_checkpoint(
            os.path.join(root_dir, str(seed), ),
            actor_net,
            value_net,
        )

    tf_agent = create_circuit_ppo_agent(
        init_train_step,
        action_tensor_spec,
        time_step_tensor_spec,
        actor_net,
        value_net,
        tf.distribute.get_strategy(),
    )
    return tf_agent.policy


def preprocess_observation(observation):
    """
    This function takes an observation from the environment and prepares it for the model inference.

    You should implement any custom preprocessing steps needed for your observations within this function.
    """

    if isinstance(observation, dict):
        time_step = ts.TimeStep(step_type=None, reward=None, discount=None,
                                observation=observation)
    else:
        time_step = observation
    return time_step


def infer_once(model, observation):
    """
    This function takes the loaded model and a preprocessed observation to perform a single inference.

    You should implement any additional logic required for your model's inference within this function.
    """
    action_step = model.action(time_step=observation)
    action = tf.nest.map_structure(lambda t: tf.squeeze(t, axis=0), action_step.action)
    return action


if __name__ == '__main__':
    def main(_):
        model = load_model()
        env = gym.make('CircuitTraining-v0')
        obs = env.observation_space.sample()
        obs = preprocess_observation(obs)
        infer_once(model, obs)


    app.run(main)
