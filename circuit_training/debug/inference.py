import os

import gym
import tensorflow as tf
from absl import app
from tf_agents.train.utils import spec_utils, train_utils, strategy_utils
from tf_agents.trajectories import time_step as ts

from learning import static_feature_cache
from learning.agent import create_circuit_ppo_agent
from learning.train_ppo import try_load_checkpoint
from model import create_models_lib
from absl import flags

delattr(flags.FLAGS, 'plc_wrapper_main')
from rl_perf.domains import circuit_training


def load_model():
    """This function loads and prepares the model for inference."""

    root_dir = os.environ['ROOT_DIR']
    seed = os.environ['GLOBAL_SEED']

    train_step = train_utils.create_train_step()
    env = gym.make('CircuitTraining-v0')
    observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
        spec_utils.get_tensor_specs(env))
    static_features = env.wrapped_env().get_static_obs()
    cache = static_feature_cache.StaticFeatureCache()
    cache.add_static_feature(static_features)

    strategy = strategy_utils.get_strategy(
        strategy_utils.TPU.value, strategy_utils.USE_GPU.value
    )
    use_model_tpu = bool(strategy_utils.TPU.value)

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
            root_dir,
            actor_net,
            value_net,
        )

    tf_agent = create_circuit_ppo_agent(
        train_step,
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
    time_step = ts.TimeStep(step_type=ts.StepType.FIRST, reward=0.0, discount=1.0, observation=observation)

    # Convert the single timestep into a batch of size 1
    time_step = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), time_step)

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
