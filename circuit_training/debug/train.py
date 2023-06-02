import os
import subprocess


def run_command(command: str, output_file: str):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(f"script_dir: {script_dir}")

    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{script_dir}:{'/rl-perf'}:{current_pythonpath}"
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    with open(output_file, "w") as outfile:
        # Pass the updated environment variables to the subprocess
        process = subprocess.Popen(command, shell=True, stdout=outfile, stderr=subprocess.STDOUT, env=env)
    return process


def train():
    root_dir = os.getenv("ROOT_DIR")
    if root_dir is None:
        raise ValueError("Missing environment variable: ROOT_DIR")
    os.makedirs(root_dir, exist_ok=True)

    num_collect_jobs = os.getenv("NUM_COLLECT_JOBS")
    if num_collect_jobs is None:
        raise ValueError("Missing environment variable: NUM_COLLECT_JOBS")
    num_collect_jobs = int(num_collect_jobs)

    global_seed = os.getenv("GLOBAL_SEED")
    if global_seed is None:
        raise ValueError("Missing environment variable: GLOBAL_SEED")

    reverb_port = os.getenv("REVERB_PORT")
    if reverb_port is None:
        raise ValueError("Missing environment variable: REVERB_PORT")

    reverb_server = os.getenv("REVERB_SERVER")
    if reverb_server is None:
        raise ValueError("Missing environment variable: REVERB_SERVER")

    netlist_file = os.getenv("NETLIST_FILE")
    if netlist_file is None:
        raise ValueError("Missing environment variable: NETLIST_FILE")

    init_placement = os.getenv("INIT_PLACEMENT")
    if init_placement is None:
        raise ValueError("Missing environment variable: INIT_PLACEMENT")

    output_dir = f"{root_dir}/output"
    os.makedirs(output_dir, exist_ok=True)

    # Start reverb server
    reverb_command = f"""
    python3.9 -m learning.ppo_reverb_server \
    --root_dir={root_dir} \
    --global_seed={global_seed} \
    --port={reverb_port} 
    """
    run_command(reverb_command, f"{output_dir}/reverb_server_output")

    train_command = f"""
    python3.9 -m learning.train_ppo \
    --root_dir={root_dir} \
    --std_cell_placer_mode=dreamplace \
    --replay_buffer_server_address={reverb_server} \
    --variable_container_server_address={reverb_server} \
    --sequence_length=134 \
    --gin_bindings='train.num_iterations=200' \
    --netlist_file={netlist_file} \
    --init_placement={init_placement} \
    --global_seed={global_seed} \
    --use_gpu
    """

    # train_command = f"""
    # python3.9 -m learning.train_ppo \
    # --root_dir={root_dir} \
    # --std_cell_placer_mode=dreamplace \
    # --replay_buffer_server_address={reverb_server} \
    # --variable_container_server_address={reverb_server} \
    # --sequence_length=3 \
    # --gin_bindings='train.num_iterations=200' \
    # --gin_bindings='train.num_episodes_per_iteration=32' \
    # --gin_bindings='train.per_replica_batch_size=64' \
    # --gin_bindings='CircuittrainingPPOLearner.summary_interval=12' \
    # --gin_bindings='CircuitPPOAgent.debug_summaries=True' \
    # --netlist_file={netlist_file} \
    # --init_placement={init_placement} \
    # --global_seed={global_seed} \
    # --use_gpu
    # """
    train_process = run_command(train_command, f"{output_dir}/train_job_output")

    # Start collect jobs
    for i in range(num_collect_jobs):
        collect_command = f"""
        CUDA_VISIBLE_DEVICES=-1 python3.9 -m learning.ppo_collect \
        --root_dir={root_dir} \
        --std_cell_placer_mode=dreamplace \
        --replay_buffer_server_address={reverb_server} \
        --variable_container_server_address={reverb_server} \
        --task_id={i} \
        --netlist_file={netlist_file} \
        --init_placement={init_placement} \
        --global_seed={global_seed} \
        --logtostderr
        """
        run_command(collect_command, f"{output_dir}/collect_job_{i:02d}")

    # Start eval job
    eval_command = f"""
    CUDA_VISIBLE_DEVICES=-1 python3.9 -m learning.eval \
    --root_dir={root_dir} \
    --variable_container_server_address={reverb_server} \
    --netlist_file={netlist_file} \
    --init_placement={init_placement} \
    --global_seed={global_seed} 
    """
    run_command(eval_command, f"{output_dir}/eval_job")
    train_process.wait()  # wait to keep process alive while collecting metrics


if __name__ == '__main__':
    train()
