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
    visualize = os.environ['VISUALIZE']
    int_save_freq = int(os.environ['INT_SAVE_FREQ'])
    setup_path = os.environ['SETUP_PATH']
    motion_file_path = os.environ['MOTION_FILE_PATH']
    output_dir = root_dir
    total_timesteps = total_timesteps // parallel_cores

    print("root_dir:", root_dir)
    print("seed:", seed)
    print("total_timesteps:", total_timesteps)
    print("parallel_mode:", parallel_mode)
    print("parallel_cores:", parallel_cores)
    print("mode:", mode)
    print("visualize:", visualize)
    print("int_save_freq:", int_save_freq)
    print("setup_path:", setup_path)
    print("output_dir:", output_dir)

    mpi_command = f"mpiexec -n {parallel_cores} python3.7 {setup_path} --mode {mode} --int_save_freq {int_save_freq} --output_dir {output_dir} --seed {seed} --total_timesteps {total_timesteps} --int_save_freq {int_save_freq} --motion_file_path {motion_file_path}"
    subprocess.run(mpi_command, shell=True, check=True)


def main(_):
    train()


if __name__ == '__main__':
    app.run(main)
