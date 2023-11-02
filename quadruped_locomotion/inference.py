import gym
from rl_perf.domains import quadruped_locomotion
import os
import tensorflow as tf
from rl_perf.domains.quadruped_locomotion.motion_imitation.learning import ppo_imitation, imitation_policies


def load_model(env):
    root_dir = os.environ['ROOT_DIR']
    seed = int(os.environ['SEED'])
    print("root_dir:", root_dir)
    print("seed:", seed)

    policies_dir = os.path.join(root_dir, 'policies')
    policy_filename = f'rl_policy_9991750_steps.zip'
    policy_path = os.path.join(policies_dir, policy_filename)

    policy_kwargs = {
        "net_arch": [{"pi": [512, 256],
                      "vf": [512, 256]}],
        "act_fun": tf.nn.relu
    }
    timesteps_per_actorbatch = 4096
    rank = 1
    optim_batchsize = 256
    model = ppo_imitation.PPOImitation(
        policy=imitation_policies.ImitationPolicy,
        env=env,
        gamma=0.95,
        timesteps_per_actorbatch=timesteps_per_actorbatch,
        clip_param=0.2,
        optim_epochs=1,
        optim_stepsize=1e-5,
        optim_batchsize=optim_batchsize,
        lam=0.95,
        adam_epsilon=1e-5,
        full_tensorboard_log=rank == 0,
        schedule='constant',
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log_dir if rank == 0 else None,
        verbose=2 * (rank == 0),
    )
    model.load(policy_path)
    return model


def infer_once(model, observation):
    # the model is from stable baselines so use it to run inference on a single observation
    action, _states = model.predict(observation)
    return action


def preprocess_observation(observation):
    return observation


def main(_):
    pass


if __name__ == '__main__':
    app.run(main)
