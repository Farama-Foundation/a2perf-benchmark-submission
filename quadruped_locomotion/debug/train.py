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
    parallel = True,
    parallel_cores = 8,
    timesteps_per_actorbatch = 4096, 
    optim_batchsize = 256,
    mode = "train", 
    total_timesteps = 2e8, 
    output_dir = "output",
    visualize = False, 
    int_save_freq = 10000000, 
    setup_path = None,
    ):
    
    if parallel:
             
        try:
            mpi_command = f"mpiexec -n {parallel_cores} python {setup_path} --mode {mode} --int_save_freq {int_save_freq} --output_dir {output_dir} --seed {seed} --total_timesteps {total_timesteps} --int_save_freq {int_save_freq} {'--visualize' if visualize else ''}"
            subprocess.run(mpi_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
        
    else: 
    
        pass

        
    
def train():
    gin.parse_config_file('./train.gin')
    seed = int(os.environ['SEED'])
    root_dir = os.environ['ROOT_DIR']
    output_dir = os.path.join(root_dir, 'policies')          
        
    train_eval(root_dir=root_dir,
               seed=seed,
               output_dir=output_dir
               )
    
if __name__ == '__main__':
    train()