import gym
from rl_perf.domains import quadruped_locomotion


def load_model():

    pass



def preprocess_observation(observation):
    pass


def main(_):
    root_dir = os.environ['ROOT_DIR']
    seed = int(os.environ['SEED'])
    # total_timesteps = int(os.environ['TOTAL_ENV_STEPS'])
    # parallel_mode = os.environ['PARALLEL_MODE']
    # parallel_cores = int(os.environ['PARALLEL_CORES'])
    # mode = os.environ['MODE']
    # visualize = bool(os.environ['VISUALIZE'])
    # int_save_freq = int(os.environ['INT_SAVE_FREQ'])
    # setup_path = os.environ['SETUP_PATH']
    # motion_file_path = os.environ['MOTION_FILE_PATH']
    # timesteps_per_actorbatch = int(np.ceil(float(TIMESTEPS_PER_ACTORBATCH) / parallel_cores))
    # optim_batchsize = int(np.ceil(float(OPTIM_BATCHSIZE) / parallel_cores))
    # output_dir = root_dir
    # print("root_dir:", root_dir)
    # print("seed:", seed)
    # print("total_timesteps:", total_timesteps)
    # print("parallel_mode:", parallel_mode)
    # print("parallel_cores:", parallel_cores)
    # print("mode:", mode)
    # print("visualize:", visualize)
    # print("int_save_freq:", int_save_freq)
    # print("setup_path:", setup_path)
    # print("output_dir:", output_dir)
    # print("timesteps_per_actorbatch:", timesteps_per_actorbatch)
    # print("optim_batchsize:", optim_batchsize)


if __name__ == '__main__':
    app.run(main)
