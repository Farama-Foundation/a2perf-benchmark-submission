import os
import subprocess


def run_command(command: str, output_file: str):
    with open(output_file, "w") as outfile:
        process = subprocess.Popen(command, shell=True, stdout=outfile, stderr=subprocess.STDOUT)


def train():
    root_dir = os.getenv("ROOT_DIR")
    if root_dir is None:
        raise ValueError("Missing environment variable: ROOT_DIR")
    os.makedirs(root_dir, exist_ok=True)

    num_collect_jobs = int(os.getenv("NUM_CT_COLLECT_JOBS", 0))

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
    CUDA_VISIBLE_DEVICES=-1 python3.9 -m learning.ppo_reverb_server \
    --root_dir={root_dir} \
    --global_seed={global_seed} \
    --port={reverb_port} \
    """
    run_command(reverb_command, f"{output_dir}/reverb_server_output")

    raise ValueError("Stop here")
    # Start train job
    train_command = f"""
    python3.9 -m learning.train_ppo \
    --root_dir={root_dir} \
    --replay_buffer_server_address={reverb_server} \
    --variable_container_server_address={reverb_server} \
    --num_episodes_per_iteration=16 \
    --global_batch_size=64 \
    --netlist_file={netlist_file} \
    --global_seed={global_seed} \
    --init_placement={init_placement}
    """
    run_command(train_command, f"{output_dir}/train_job_output")

    # Start collect jobs
    for i in range(num_collect_jobs):
        collect_command = f"""
        CUDA_VISIBLE_DEVICES=-1 python3.9 -m learning.ppo_collect \
        --root_dir={root_dir} \
        --replay_buffer_server_address={reverb_server} \
        --variable_container_server_address={reverb_server} \
        --task_id={i} \
        --netlist_file={netlist_file} \
        --global_seed={global_seed} \
        --init_placement={init_placement}
        """
        run_command(collect_command, f"{output_dir}/collect_job_{i:02d}")

    # Start eval job
    eval_command = f"""
    CUDA_VISIBLE_DEVICES=-1 python3.9 -m learning.eval \
    --root_dir={root_dir} \
    --variable_container_server_address={reverb_server} \
    --netlist_file={netlist_file} \
    --global_seed={global_seed} \
    --init_placement={init_placement}
    """
    run_command(eval_command, f"{output_dir}/eval_job")


if __name__ == '__main__':
    train()

# Here are the commands from the circuit training example

# Training job
# $ docker run --network host -d -e "GOOGLE_APPLICATION_CREDENTIALS=/workspace/cloud_key.json" \
#      --gpus all  --rm -it -v ${REPO_ROOT}:/workspace -w /workspace/ circuit_training:core  \
#      python3.9 -m circuit_training.learning.train_ppo \
#        --root_dir=${ROOT_DIR} \
#        --std_cell_placer_mode=dreamplace \
#        --replay_buffer_server_address=${REVERB_SERVER} \
#        --variable_container_server_address=${REVERB_SERVER} \
#        --sequence_length=134 \
#        --gin_bindings='train.num_iterations=200'\
#        --netlist_file=${NETLIST_FILE} \
#        --init_placement=${INIT_PLACEMENT} \
#        --global_seed=${GLOBAL_SEED} \
#        --use_gpu
#
# # If using the toy netlist, some args need changed. Use this command instead.
# $ docker run --network host -d -e "GOOGLE_APPLICATION_CREDENTIALS=/workspace/cloud_key.json" \
#      --gpus all  --rm -it -v ${REPO_ROOT}:/workspace -w /workspace/ circuit_training:core  \
#      python3.9 -m circuit_training.learning.train_ppo \
#        --root_dir=${ROOT_DIR} \
#        --std_cell_placer_mode=dreamplace \
#        --replay_buffer_server_address=${REVERB_SERVER} \
#        --variable_container_server_address=${REVERB_SERVER} \
#        --sequence_length=3 \
#        --gin_bindings='train.num_iterations=200' \
#        --gin_bindings='train.num_episodes_per_iteration=32' \
#        --gin_bindings='train.per_replica_batch_size=64' \
#        --gin_bindings='CircuittrainingPPOLearner.summary_interval=12' \
#        --gin_bindings='CircuitPPOAgent.debug_summaries=True' \
#        --netlist_file=${NETLIST_FILE} \
#        --init_placement=${INIT_PLACEMENT} \
#        --global_seed=${GLOBAL_SEED} \
#        --use_gpu


# Collect job
#
# for i in $(seq 1 23); do
#   docker run --network host -d -e "GOOGLE_APPLICATION_CREDENTIALS=/workspace/cloud_key.json" \
#   --rm -it -v ${REPO_ROOT}/circuit_training:/workspace -w /workspace/ circuit_training:core  \
#      python3.9 -m circuit_training.learning.ppo_collect \
#   --root_dir=${ROOT_DIR} \
#   --std_cell_placer_mode=dreamplace \
#   --replay_buffer_server_address=${REVERB_SERVER} \
#   --variable_container_server_address=${REVERB_SERVER} \
#   --task_id=${i} \
#   --netlist_file=${NETLIST_FILE} \
#   --init_placement=${INIT_PLACEMENT} \
#   --global_seed=${GLOBAL_SEED} \
#   --logtostderr


# Eval job
# $ docker run --network host -d -e "GOOGLE_APPLICATION_CREDENTIALS=/workspace/cloud_key.json" \
#      --rm -it -v $(pwd):/workspace -w /workspace/ circuit_training:core  \
#      python3.9 -m circuit_training.learning.eval \
#        --root_dir=${ROOT_DIR} \
#        --variable_container_server_address=${REVERB_SERVER} \
#        --netlist_file=${NETLIST_FILE} \
#        --init_placement=${INIT_PLACEMENT} \
#        --global_seed=${GLOBAL_SEED} \
#        --output_placement_save_dir=./
