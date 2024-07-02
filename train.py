import os
import time

import gin
from absl import logging

from commands import collect_command
from commands import create_and_manage_process
from commands import reverb_command
from commands import train_command

PROCESS_WAIT_INTERVAL = 10


@gin.configurable
def train_func(
    # Basic configuration
    task_name: str = None,
    algorithm: str = None,
    seed: int = -1,
    root_dir: str = "/experiment_dir",
    debug: bool = False,
    # Environment configuration
    env_name: str = None,
    env_batch_size: int = -1,
    max_sequence_length: int = -1,
    num_episodes_per_iteration: int = -1,
    # Training hyperparameters
    num_iterations: int = -1,
    num_epochs: int = -1,
    batch_size: int = -1,
    learning_rate: float = -1.0,
    use_gae: bool = False,
    entropy_regularization: float = -1.0,
    exploration_noise_std: float = -1.0,
    epsilon_greedy: float = -1.0,
    # Replay buffer configuration
    replay_buffer_capacity: int = -1,
    # Logging and checkpointing
    eval_interval: int = -1,
    train_checkpoint_interval: int = -1,
    policy_checkpoint_interval: int = -1,
    log_interval: int = -1,
    # Distributed training
    num_replicas: int = -1,
    num_collect_jobs: int = -1,
    ## Environment-specific parameters
    # WebNavigation-v0
    difficulty_level: int = -1,
    num_websites: int = -1,
    max_vocab_size: int = -1,
    embedding_dim: int = -1,
    latent_dim: int = -1,
    profile_value_dropout: float = -1.0,
    # CircuitTraining-v0
    netlist_path: str = None,
    init_placement_path: str = None,
    std_cell_placer_mode: str = None,
    # QuadrupedLocomotion-v0
    motion_file_path: str = None,
    # Distributed training configurations
    vocab_port: int = 50000,
    vocabulary_manager_auth_key: str = "",
    replay_buffer_server_address: str = None,
    variable_container_server_address: str = None,
    replay_buffer_server_port: int = -1,
    variable_container_server_port: int = -1,
    vocabulary_server_address: str = None,
    vocabulary_server_port: int = -1,
    # Minari configurations
    dataset_id: str = None,
):
    # Check if the selected algorithm is Proximal Policy Optimization (PPO)
    if algorithm in ("ppo",):
        train_steps_per_iteration = max(
            1,
            int(
                num_episodes_per_iteration
                * max_sequence_length
                / batch_size
                * num_epochs
                / num_replicas
            ),
        )

        # Shuffle three episodes worth of samples
        shuffle_buffer_size = 3

        # Only a single iteration is performed per call to the learner. We set the
        # `num_samples` argument to `env_batch_size` to ensure that the learner
        # processes all the data collected by the actors in a single call.
        learner_iterations_per_call = 1

        # No need to collect data initially in PPO.
        initial_collect_steps = 0

        min_table_size_before_sampling = 1

    elif algorithm in ("sac", "ddqn", "td3", "ddpg", "dqn"):
        learner_iterations_per_call = 1
        train_steps_per_iteration = 1
        shuffle_buffer_size = -1
        initial_collect_steps = max_sequence_length
        min_table_size_before_sampling = 1
    elif algorithm in ("bc",):
        learner_iterations_per_call = 1
        train_steps_per_iteration = 1
        shuffle_buffer_size = -1
        initial_collect_steps = 0
        min_table_size_before_sampling = 1
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if train_steps_per_iteration < 1:
        raise ValueError(
            "train_steps_per_iteration must be at least 1, got"
            f" {train_steps_per_iteration}"
        )

    # All intervals are given in terms of iterations, so translate them to train steps
    log_interval *= train_steps_per_iteration
    train_checkpoint_interval *= train_steps_per_iteration
    policy_checkpoint_interval *= train_steps_per_iteration
    print("Computed train_steps_per_iteration:", train_steps_per_iteration)
    print("Computed log_interval:", log_interval)
    print("Computed train_checkpoint_interval:", train_checkpoint_interval)
    print("Computed policy_checkpoint_interval:", policy_checkpoint_interval)

    max_train_steps = train_steps_per_iteration * num_iterations
    print("shuffle_buffer_size:", shuffle_buffer_size)
    print("train_steps_per_iteration:", train_steps_per_iteration)
    print("learner_iterations_per_call:", learner_iterations_per_call)
    print("initial_collect_steps:", initial_collect_steps)
    print("num_iterations:", num_iterations)
    print("min_table_size_before_sampling:", min_table_size_before_sampling)
    print("max_train_steps:", max_train_steps)

    all_processes = []
    env_flags = []
    if env_name == "WebNavigation-v0":
        env_flags.extend(
            [
                f"--env_name={env_name}",
                f"--num_websites={num_websites}",
                f"--difficulty_level={difficulty_level}",
                f"--profile_value_dropout={profile_value_dropout}",
                f"--embedding_dim={embedding_dim}",
                f"--latent_dim={latent_dim}",
                f"--max_vocab_size={max_vocab_size}",
            ]
        )

        # Start vocabulary manager for WebNavigation-v0
        vocab_manager_command = [
            "python",
            "train_lib/vocabulary_manager.py",
            f"--vocabulary_server_port={vocab_port}",
            f"--vocabulary_server_address={vocabulary_server_address}",
            f"--vocabulary_manager_auth_key={vocabulary_manager_auth_key}",
            f"--max_vocab_size={max_vocab_size}",
            f"--verbosity={logging.get_verbosity()}",
            f"--root_dir={root_dir}",
        ]
        print("Command for vocab manager:", vocab_manager_command)
        vocab_server_process = create_and_manage_process(
            vocab_manager_command, all_processes
        )

        if vocab_server_process.poll() is not None:
            raise ValueError("Vocabulary manager server failed to start.")
        else:
            print("Successfully launched vocab manager server.")

    elif env_name == "QuadrupedLocomotion-v0":
        env_flags.extend(
            [f"--env_name={env_name}", f"--motion_file_path={motion_file_path}"]
        )
    elif env_name == "CircuitTraining-v0":
        env_flags.extend(
            [
                f"--std_cell_placer_mode={std_cell_placer_mode}",
                f"--netlist_file={netlist_path}",
                f"--init_placement={init_placement_path}",
            ]
        )
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    # Start collect jobs (if not using BC)
    if algorithm != "bc":
        collect_job_commands = [
            collect_command(
                algorithm=algorithm,
                env_name=env_name,
                debug=debug,
                env_batch_size=env_batch_size,
                num_iterations=num_iterations,
                initial_collect_steps=initial_collect_steps,
                max_sequence_length=max_sequence_length,
                epsilon_greedy=epsilon_greedy,
                max_train_steps=max_train_steps,
                num_replicas=num_replicas,
                replay_buffer_server_address=replay_buffer_server_address,
                replay_buffer_server_port=replay_buffer_server_port,
                root_dir=root_dir,
                seed=seed,
                log_interval=log_interval,
                variable_container_server_address=variable_container_server_address,
                variable_container_server_port=variable_container_server_port,
                vocabulary_manager_auth_key=vocabulary_manager_auth_key,
                vocabulary_server_address=vocabulary_server_address,
                vocabulary_server_port=vocabulary_server_port,
                task=i,
            )
            + env_flags
            for i in range(num_collect_jobs)
        ]

        print("Collect job command (example):", " ".join(collect_job_commands[0]))

        collect_jobs = []
        for command in collect_job_commands:
            collect_jobs.append(create_and_manage_process(command, all_processes))
        print("Successfully launched collect jobs.")

    # Start reverb server (if not using BC)
    if algorithm != "bc":
        reverb_job_command = reverb_command(
            task_name=task_name,
            replay_buffer_server_port=replay_buffer_server_port,
            root_dir=root_dir,
            replay_buffer_capacity=replay_buffer_capacity,
            algorithm=algorithm,
            min_table_size_before_sampling=min_table_size_before_sampling,
        )
        print("Reverb job command:", " ".join(reverb_job_command))
        create_and_manage_process(
            reverb_job_command,
            all_processes,
            env_vars=dict(**os.environ, CUDA_VISIBLE_DEVICES="-1"),
        )
        print("Successfully launched reverb server.")

    # Start train job
    train_job_command = (
        train_command(
            num_iterations=num_iterations,
            entropy_regularization=entropy_regularization,
            use_gae=use_gae,
            root_dir=root_dir,
            env_name=env_name,
            dataset_id=dataset_id,
            vocabulary_manager_auth_key=vocabulary_manager_auth_key,
            vocabulary_server_port=vocab_port,
            vocabulary_server_address=vocabulary_server_address,
            variable_container_server_address=variable_container_server_address,
            variable_container_server_port=variable_container_server_port,
            replay_buffer_server_address=replay_buffer_server_address,
            replay_buffer_server_port=replay_buffer_server_port,
            max_sequence_length=max_sequence_length,
            num_episodes_per_iteration=num_episodes_per_iteration,
            log_interval=log_interval,
            task_name=task_name,
            use_gpu=True,
            seed=seed,
            num_epochs=num_epochs,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_replicas=num_replicas,
            algorithm=algorithm,
            debug=debug,
            epsilon_greedy=epsilon_greedy,
            train_checkpoint_interval=train_checkpoint_interval,
            policy_checkpoint_interval=policy_checkpoint_interval,
            env_batch_size=env_batch_size,
            learning_rate=learning_rate,
            exploration_noise_std=exploration_noise_std,
            max_train_steps=max_train_steps,
            learner_iterations_per_call=learner_iterations_per_call,
        )
        + env_flags
    )

    print("Train job command:", " ".join(train_job_command))
    create_and_manage_process(train_job_command, all_processes)
    print("Successfully launched train job.")

    # Wait for training to complete
    while not os.path.exists(os.path.join(root_dir, "training_complete")):
        time.sleep(PROCESS_WAIT_INTERVAL)

    print("Training complete.")


def train(
    gin_config_path: str,
):
    gin.parse_config_file(gin_config_path)
    train_func()
