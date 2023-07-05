import os
import subprocess


def train():
    root_dir = os.getenv("ROOT_DIR")
    if root_dir is None:
        raise ValueError("Missing environment variable: ROOT_DIR")
    os.makedirs(root_dir, exist_ok=True)

    num_collect_jobs = os.getenv("NUM_COLLECT_JOBS")
    if num_collect_jobs is None:
        raise ValueError("Missing environment variable: NUM_COLLECT_JOBS")

    global_seed = os.getenv("GLOBAL_SEED")
    if global_seed is None:
        raise ValueError("Missing environment variable: GLOBAL_SEED")

    reverb_port = os.getenv("REVERB_PORT")
    if reverb_port is None:
        raise ValueError("Missing environment variable: REVERB_PORT")

    reverb_server_ip = os.getenv("REVERB_SERVER_IP")
    if reverb_server_ip is None:
        raise ValueError("Missing environment variable: REVERB_SERVER_IP")

    netlist_file = os.getenv("NETLIST_FILE")
    if netlist_file is None:
        raise ValueError("Missing environment variable: NETLIST_FILE")

    init_placement = os.getenv("INIT_PLACEMENT")
    if init_placement is None:
        raise ValueError("Missing environment variable: INIT_PLACEMENT")

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["WRAPT_DISABLE_EXTENSIONS"] = "true"
    process = subprocess.Popen("bash train.sh", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    process.wait()


if __name__ == '__main__':
    train()
