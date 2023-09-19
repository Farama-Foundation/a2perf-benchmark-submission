import gym
import time
import gin
import os
import subprocess


@gin.configurable
def train_eval(
        root_dir,
        seed=0,
        env_name='',
        parallel_mode=True,
        parallel_cores=8,
        timesteps_per_actorbatch=4096,
        optim_batchsize=256,
        mode="train",
        total_timesteps=2e8,
        motion_file_path=None,
        output_dir="output",
        visualize=False,
        int_save_freq=10000000,
        setup_path=None,
):
    if parallel_mode:

        try:
            # mpi_command = f"mpiexec -n {parallel_cores} python {setup_path} --mode {mode} --int_save_freq {int_save_freq} --output_dir {output_dir} --seed {seed} --total_timesteps {total_timesteps} --int_save_freq {int_save_freq} {'--visualize' if visualize else ''}"
            mpi_command = f"mpiexec -n {parallel_cores} python {setup_path}" \
                          f" --mode {mode}" \
                          f" --int_save_freq {int_save_freq}" \
                          f" --output_dir {output_dir}" \
                          f" --seed {seed}" \
                          f" --total_timesteps {total_timesteps}" \
                          f" --motion_file_path {motion_file_path}"

            subprocess.run(mpi_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")

    else:

        pass


def train():
    gin.parse_config_file('./train.gin')
    root_dir = os.environ['ROOT_DIR']
    seed = int(os.environ['SEED'])
    total_timesteps = int(os.environ['TOTAL_ENV_STEPS'])
    parallel_mode = os.environ['PARALLEL_MODE']
    parallel_cores = int(os.environ['PARALLEL_CORES'])
    mode = os.environ['MODE']
    visualize = os.environ['VISUALIZE']
    int_save_freq = int(os.environ['INT_SAVE_FREQ'])
    setup_path = os.environ['SETUP_PATH']
    output_dir = os.path.join(root_dir, 'policies')
    motion_file_path = os.environ['MOTION_FILE_PATH']
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

    train_eval(root_dir=root_dir,
               seed=seed,
               output_dir=output_dir,
               total_timesteps=total_timesteps,
               parallel_mode=parallel_mode,
               parallel_cores=parallel_cores,
               mode=mode,
               visualize=visualize,
               int_save_freq=int_save_freq,
               setup_path=setup_path
               ,
               motion_file_path=motion_file_path
               )


if __name__ == '__main__':
    train()
