import gym
import time
import gin
import os
import subprocess
from rl_perf.domains import quadruped_locomotion

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
    total_timesteps = 100000000, 
    output_dir = "output",
    visualize = False, 
    motion_file = "motion_imitation/data/motions/dog_pace.txt", 
    int_save_freq = 10000000, 
    env_path = None,
    ):
    
    if parallel:
             
        try:
            mpi_command = f"mpiexec -n {parallel_cores} python {env_path} --mode {mode} --motion_file {motion_file} --int_save_freq {int_save_freq} --output_dir {output_dir}"
            subprocess.run(mpi_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
        
    else: 
    
        #TODO: PASS IN PARAMETERS
        quadruped_locomotion_env = gym.make(env_name)
        
        # Set Seed
        quadruped_locomotion_env.set_rand_seed(seed)
        
        #Make Environment
        env = quadruped_locomotion_env.build_environment()
        
        # Initialize Built in Model
        model = quadruped_locomotion_env.build_model(env, timesteps_per_actorbatch, optim_batchsize)
        
        # Train Model
        quadruped_locomotion_env.train(model, env)
        
    
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