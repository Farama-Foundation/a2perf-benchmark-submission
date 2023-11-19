import numpy as np
import os
import subprocess
from absl import app

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256


def train():
    root_dir = os.environ['ROOT_DIR']
    seed = int(os.environ['SEED'])
    total_timesteps = int(os.environ['TOTAL_ENV_STEPS'])
    parallel_mode = os.environ['PARALLEL_MODE']
    parallel_cores = int(os.environ['PARALLEL_CORES'])
    mode = os.environ['MODE']
    visualize = bool(os.environ['VISUALIZE'])
    int_save_freq = int(os.environ['INT_SAVE_FREQ'])
    int_eval_freq = int(os.environ['INT_EVAL_FREQ'])
    setup_path = os.environ['SETUP_PATH']
    motion_file_path = os.environ['MOTION_FILE_PATH']
    timesteps_per_actorbatch = int(np.ceil(float(TIMESTEPS_PER_ACTORBATCH) / parallel_cores))
    optim_batchsize = int(np.ceil(float(OPTIM_BATCHSIZE) / parallel_cores))
    output_dir = root_dir
    print("root_dir:", root_dir)
    print("seed:", seed)
    print("total_timesteps:", total_timesteps)
    print("parallel_mode:", parallel_mode)
    print("parallel_cores:", parallel_cores)
    print("mode:", mode)
    print("visualize:", visualize)
    print("int_save_freq:", int_save_freq)
    print("int_eval_freq:", int_eval_freq)
    print("setup_path:", setup_path)
    print("output_dir:", output_dir)
    print("timesteps_per_actorbatch:", timesteps_per_actorbatch)
    print("optim_batchsize:", optim_batchsize)

    mpi_command = f'mpiexec -n {parallel_cores}' \
                  f' python3.9 {setup_path}' \
                  f' --mode {mode}' \
                  f' --int_save_freq {int_save_freq}' \
                  f' --int_eval_freq {int_eval_freq}' \
                  f' --output_dir {output_dir}' \
                  f' --seed {seed}' \
                  f' --total_timesteps {total_timesteps}' \
                  f' {"--visualize" if visualize == "True" else ""}' \
                  f' --total_timesteps {total_timesteps}' \
                  f' --motion_file_path {motion_file_path}' \
                  f' --timesteps_per_actorbatch {timesteps_per_actorbatch}' \
                  f' --optim_batchsize {optim_batchsize}'

    print("mpi_command:", mpi_command)
    subprocess.run(mpi_command, shell=True, check=True)


def main(_):
    train()


if __name__ == '__main__':
    app.run(main)
