import gym
import subprocess
from pathlib import Path
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from schedule import LinearExploration, LinearSchedule
from hydra import Hydra

#from configs.hydra_no_div_big_r import config
#from configs.hydra_no_div_small_r import config
#from configs.hydra_test import config
from configs.hydra import config

import time

from utils.hydra_preprocess import my_prepro

"""
Use deep Q network for the Atari game. This is an optional part of the assignment and will not be graded.
Feel free to change the configurations (in the configs/ folder). 

You'll find the results, log and video recordings of your agent every 250k under
the corresponding file in the results folder. A good way to monitor the progress
of the training is to use Tensorboard. The starter code writes summaries of different
variables.
"""
if __name__ == '__main__':
    # video problem:
    # https://github.com/openai/gym/pull/2139/commits/5c94ebabded3af1929033b72cba1c00e87c84dcf

    start = time.time()
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=my_prepro, shape=(40, 40, config.nb_pawns+1), 
                        overwrite_render=config.overwrite_render)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

#    # Check weights
#    load_path = Path(config.load_path)
#    if load_path.is_file():
#        print(f'File {load_path} exists.')
#    else:
#        print(f'You need to have the file {load_path}.')

    # train model
    model = Hydra(env, config)
    model.run(exp_schedule, lr_schedule)
#    model.run_no_train(exp_schedule, lr_schedule)
    
    print(f'Runtime {time.time()-start}s')
