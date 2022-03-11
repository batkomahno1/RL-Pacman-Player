#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:20:10 2021

@author: stierlitz
"""
#
## test stuff
#from q2_schedule import LinearExploration, LinearSchedule
#from utils.test_env import EnvTest
#from configs.q4_nature import config
#from core.deep_q_learning_torch import DQN
#
##"""
##Use deep Q network for test environment.
##"""
#if __name__ == '__main__':
##    env = EnvTest((8, 8, 6))
#    env = EnvTest((5, 5, 1))
#
#    # exploration strategy
#    exp_schedule = LinearExploration(env, config.eps_begin,
#            config.eps_end, config.eps_nsteps)
#
#    # learning rate schedule
#    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
#            config.lr_nsteps)
#
#    # train model
#    model = DQN(env, config)
#    model.run(exp_schedule, lr_schedule)

################################
################################
#
## test stuff
#from q2_schedule import LinearExploration, LinearSchedule
#from utils.test_env import EnvTest
#from configs.q4_nature import config
#from core.test_head import DQN
#
##"""
##Use deep Q network for test environment.
##"""
#if __name__ == '__main__':
##    env = EnvTest((8, 8, 6))
#    env = EnvTest((5, 5, 1))
#
#    # exploration strategy
#    exp_schedule = LinearExploration(env, config.eps_begin,
#            config.eps_end, config.eps_nsteps)
#
#    # learning rate schedule
#    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
#            config.lr_nsteps)
#
#    # train model
#    model = DQN(env, config)
#    model.run(exp_schedule, lr_schedule)
#
#################################
#################################
##

from configs.hydra import config
import time
import gym
from utils.hydra_preprocess import my_prepro, pics
from utils.wrappers import PreproWrapper, MaxAndSkipEnv
import matplotlib.pyplot as plt
import copy
from PIL import Image

if __name__ == '__main__':
    # video problem:
    # https://github.com/openai/gym/pull/2139/commits/5c94ebabded3af1929033b72cba1c00e87c84dcf
    
    start = time.time()
    # make env
    env = gym.make(config.env_name)
    env = MaxAndSkipEnv(env, skip=config.skip_frame)
    env = PreproWrapper(env, prepro=my_prepro, shape=(40, 40, 142), 
                        overwrite_render=True)
    
    state = env.reset()
#    print(state.shape)
    
    for i in range(1):
#        env.render()
#        print(state.shape, state[...,0].shape, (state[...,0]==255).any())
        img = state#[...,0]
        
#        im = Image.fromarray(np.load('output.npy')).convert('RGB')
#        im.save(f'imgs/final_scaled.png')
#        time.sleep(0.5)
#        img[img==0]=1
#        img[img==255]=0
#        img*=255

#        env._render()
#        var = copy.deepcopy(state)
#        state = my_prepro(state)
#        print(i,'\r', end='')
#        time.sleep(0.1)
        action = env.action_space.sample()
        obs = env.step(action)
        state = obs[0]
        if obs[-2]:env.reset()
#        print(obs[1])
#        if obs[3]['ale.lives'] == 0: print(obs[3])
#        if obs[2]: 
#            print(obs[1],obs[3]['ale.lives'])
#            time.sleep(1)


    
        