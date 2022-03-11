from utils.hydra_preprocess import my_prepro
import numpy as np

class config():
    # env config
    test = False
    num_cores = 10
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

    # model and training config
#    load_path         = model_output
    num_episodes_test = 5 if not test else 1
    grad_clip         = True
    clip_val          = 1e9 # TODO: grads get clipped too much
    saving_freq       = 50000
    log_freq          = 50
    eval_freq         = 50000
    record_freq       = 250000
    soft_epsilon      = 0.0

    # nature paper hyper params
    nsteps_train       = 1000000 if not test else 10000
    batch_size         = 32 if not test else 16
    buffer_size        = 200000#0
    target_update_freq = 10000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 2 #MUST BE TWO FRAMES!!
    skip_frame         = 4
    lr_begin           = 0.00008
    lr_end             = 0.00005
    lr_nsteps          = 500000
    eps_begin          = 0.5
    eps_end            = 0.05 #otherwise the pacman is in danger
    eps_nsteps         = 500000
    learning_start     = 50000 if not test else 500
    lin_output_size    = 512
    
    # my stuff
    nb_pawns = 4+4+151#133
    ghost_dist = 4 #pacman height is 2 pixels
    prepro = my_prepro
    
    pseudo_reward_pos = 100
    pseudo_reward_neg = -1
    weights = np.array([-100]*4+[100]*4+[1]*151)
    
    diversification = False
    div_steps = 100
    max_div_q_val = 20
