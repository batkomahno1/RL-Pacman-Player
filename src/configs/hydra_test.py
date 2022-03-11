from utils.hydra_preprocess import my_prepro
import numpy as np

class config():
    # env config
    render_train     = False
    render_test      = False
    env_name         = "MsPacman-v0"
    overwrite_render = True
    record           = False
    high             = 255.

    # output config
    output_path  = "results/hydra/"
    model_output = output_path + "model.weights"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"

    nsteps_train       = 1000#500000
    
    # model and training config
#    load_path         = model_output
    num_episodes_test = 1#50
    grad_clip         = False
    clip_val          = 100 # TODO: grads get clipped too much
    saving_freq       = nsteps_train//10
    log_freq          = 50
    eval_freq         = 100#000000#nsteps_train//10
    record_freq       = 500#00
    soft_epsilon      = 0.05

    # nature paper hyper params
    
    batch_size         = 32
    buffer_size        = 100000#0
    target_update_freq = nsteps_train//100
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 2 #TODO:  two frames is enough to indicicate pawn direction
    skip_frame         = 4
    lr_begin           = 0.00008
    lr_end             = 0.00005
    lr_nsteps          = nsteps_train//2
    eps_begin          = 0.5
    eps_end            = 0.1
    eps_nsteps         = nsteps_train//2
    learning_start     = 500#50000
    lin_output_size    = 512
    
    # my stuff
    nb_pawns = 4+4+133
    prepro = my_prepro
    
    reward_norm = 10
    pseudo_reward_pos = 100/reward_norm#TODO:  decrease grads size
    pseudo_reward_neg = -1/reward_norm# decrease grads size
    weights = np.array([-1000]*4+[1000]*4+[10]*133)/reward_norm# decrease grads size
    
    diversification = True
    max_div_q_val = 20/reward_norm