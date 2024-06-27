import os
import subprocess

from absl import logging


def collect_command(
    algorithm,
    debug,
    env_batch_size,
    env_name,
    initial_collect_steps,
    epsilon_greedy,
    num_iterations,
    max_sequence_length,
    max_train_steps,
    num_replicas,
    replay_buffer_server_address,
    replay_buffer_server_port,
    root_dir,
    seed,
    log_interval,
    variable_container_server_address,
    variable_container_server_port,
    vocabulary_manager_auth_key,
    vocabulary_server_address,
    vocabulary_server_port,
    task,
):
    return [
        "python",
        "-m",
        "train_lib.collect",
        f"--algorithm={algorithm}",
        f"--debug={debug}",
        f"--env_batch_size={env_batch_size}",
        f"--env_name={env_name}",
        f"--epsilon_greedy={epsilon_greedy}",
        f"--initial_collect_steps={initial_collect_steps}",
        f"--max_sequence_length={max_sequence_length}",
        f"--max_train_steps={max_train_steps}",
        f"--num_iterations={num_iterations}",
        f"--num_replicas={num_replicas}",
        f"--replay_buffer_server_address={replay_buffer_server_address}:{replay_buffer_server_port}",
        f"--root_dir={root_dir}",
        f"--seed={seed}",
        f"--summary_interval={log_interval}",
        f"--task={task}",
        f"--variable_container_server_address={variable_container_server_address}:{variable_container_server_port}",
        f'--verbosity={"1" if task == 0 else "-1"}',
        f"--vocabulary_manager_auth_key={vocabulary_manager_auth_key}",
        f"--vocabulary_server_address={vocabulary_server_address}",
        f"--vocabulary_server_port={vocabulary_server_port}",
    ]


def reverb_command(
    task_name,
    replay_buffer_server_port,
    root_dir,
    replay_buffer_capacity,
    algorithm,
    min_table_size_before_sampling,
):
    return [
        "python",
        "-m",
        "train_lib.reverb_server",
        f"--port={replay_buffer_server_port}",
        f"--task_name={task_name}",
        f"--root_dir={root_dir}",
        f"--replay_buffer_capacity={replay_buffer_capacity}",
        f"--algorithm={algorithm}",
        f"--min_table_size_before_sampling={min_table_size_before_sampling}",
        f"--verbosity={logging.get_verbosity()}",
    ]


def create_and_manage_process(command, process_list, env_vars=None):
    if env_vars is None:
        env_vars = os.environ.copy()

    process = subprocess.Popen(
        command,
        env=env_vars,
        text=True,
    )

    process_list.append(process)
    return process


def train_command(
    num_iterations,
    entropy_regularization,
    use_gae,
    root_dir,
    variable_container_server_address,
    variable_container_server_port,
    vocabulary_manager_auth_key,
    vocabulary_server_port,
    vocabulary_server_address,
    replay_buffer_server_address,
    replay_buffer_server_port,
    env_name,
    max_sequence_length,
    num_episodes_per_iteration,
    log_interval,
    use_gpu,
    seed,
    task_name,
    dataset_id,
    num_epochs,
    batch_size,
    shuffle_buffer_size,
    num_replicas,
    algorithm,
    debug,
    epsilon_greedy,
    train_checkpoint_interval,
    policy_checkpoint_interval,
    env_batch_size,
    learning_rate,
    exploration_noise_std,
    max_train_steps,
    learner_iterations_per_call,
):
    return [
        "python",
        "-m",
        "train_lib.train",
        f"--algorithm={algorithm}",
        f"--dataset_id={dataset_id}",
        f"--batch_size={batch_size}",
        f"--env_batch_size={env_batch_size}",
        f"--vocabulary_server_port={vocabulary_server_port}",
        f"--vocabulary_server_address={vocabulary_server_address}",
        f"--vocabulary_manager_auth_key={vocabulary_manager_auth_key}",
        f"--debug={debug}",
        f"--entropy_regularization={entropy_regularization}",
        f"--env_name={env_name}",
        f"--epsilon_greedy={epsilon_greedy}",
        f"--exploration_noise_std={exploration_noise_std}",
        f"--learner_iterations_per_call={learner_iterations_per_call}",
        f"--learning_rate={learning_rate}",
        f"--log_interval={log_interval}",
        f"--sequence_length={max_sequence_length}",
        f"--max_train_steps={max_train_steps}",
        f"--num_epochs={num_epochs}",
        f"--num_iterations={num_iterations}",
        f"--policy_checkpoint_interval={policy_checkpoint_interval}",
        f"--replay_buffer_server_address={replay_buffer_server_address}:{replay_buffer_server_port}",
        f"--root_dir={root_dir}",
        f"--seed={seed}",
        f"--shuffle_buffer_size={shuffle_buffer_size}",
        f"--num_episodes_per_iteration={num_episodes_per_iteration}",
        f"--train_checkpoint_interval={train_checkpoint_interval}",
        f"--use_gae={use_gae}",
        f"--use_gpu={use_gpu}",
        f"--task_name={task_name}",
        f"--variable_container_server_address={variable_container_server_address}:{variable_container_server_port}",
        # Only use these if you have a pretrained policy to bootstrap from
        # f'--policy_saved_model_dir={root_dir}/policies/policy',
        # f'--policy_checkpoint_dir={root_dir}/policies/checkpoints',
    ]
