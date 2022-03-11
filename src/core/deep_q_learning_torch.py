import torch
import numpy as np
import torch.nn as nn

from typing import Tuple
from pathlib import Path
from torch.tensor import Tensor
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from core.q_learning import QN

import copy

from PIL import Image
import time
import pickle

# parallelism
import ipyparallel as ipp
from functools import partial
import os
import subprocess

def centroid(x, y):
    if x.shape[0]==0 or y.shape[0]==0: return np.array([])
    assert x.shape[0] == y.shape[0]
    mid = lambda v: v.min()+(v.max() - v.min())//2
    return np.array([mid(x), mid(y)])

def parallel_func(sp_batch, r_batch, pseudo_reward_pos, pellets, ch_len, debug, ghost_dist, row):
    rewards, done = np.zeros((ch_len)), np.zeros((ch_len))
    
    if debug: 
        var = np.zeros((40, 40,3))+255              
        for k in range(1, ch_len): var[np.where(sp_batch[row, :, :, k] > 0)] = [0, 255, 0]
        var[np.where(sp_batch[row, :, :, 0] > 0)] = [255, 0, 0]
        
    # checks for min distance from ghost
    # for first frame frame
    # ghosts
    col = -1
    # check if pacman is on screen
    if (sp_batch[row, :, :, 0] > 0).any():
        pacman = np.where(sp_batch[row, :, :, 0] > 0)
        c1 = centroid(*pacman)
        
        for i in range(1, 1+4+4):
            ghost = np.where(sp_batch[row, :, :, i] > 0)
            if ghost[0].shape[0] != 0:
                c2 = centroid(*ghost)
                dist = np.linalg.norm(c2-c1)
                if dist < ghost_dist:
                    if debug: var[ghost] = [0, 0, 255]
                    col = i
                    rewards[col-1] = pseudo_reward_pos
                    done[col-1] = 1
        
        # pellets
        # must testing for game reward because detection sometimes makes big pacman boxes
        if (r_batch[row] == 10).any() and col == -1:
            pacman = set(zip(*np.where(sp_batch[row, :, :, 0] > 0)))
            min_dist = np.inf
            for i in range(1+4+4, ch_len + 1):
                pellet = np.where(pellets[:, :, i-(1+4+4)] > 0)
                c2 = centroid(*pellet)
                if pellet[0].shape[0] != 0:
                    c2 = centroid(*pellet)
                    dist = np.linalg.norm(c2-c1)
                    if dist < min_dist:
                        min_dist = dist
                        col = i
            rewards[col-1] = pseudo_reward_pos
            done[col-1] = 1

            if debug: var[np.where(pellets[:, :, col-1-4-4] > 0)] = [0, 0, 255]                        

        if debug:
            im = Image.fromarray(var.astype(np.uint8)).convert('RGB')
            im.save(f'imgs/img_array{time.time()}.png')
    return rewards, done

def get_directions(states, ch_len, row):
    # four possible directions: tsae, tsew, htron, htous
    nb_objects = 9
    dirn = np.zeros((nb_objects, 4))
    # do just pacman and the ghosts
    for i in range(nb_objects):
        if (states[row, ..., i] > 0).any() and (states[row, ..., i + ch_len] > 0).any():
            # first frame
            cent1 = centroid(*np.where(states[row, ..., i] > 0))
            #second frame
            cent2 = centroid(*np.where(states[row, ..., i + ch_len] > 0))
            if cent1.shape[0] >= cent2.shape[0] > 0:
                x1,y1 = cent1
                x2,y2 = cent2
                # get max displacement
                if np.abs(x2-x1) > np.abs(y2-y1):
                    if x1 > x2:
                        dirn[i, 0] = 1
                    else:
                        dirn[i, 1] = 1
                else:
                    if y1 > y2:
                        dirn[i, 2] = 1
                    else:
                        dirn[i, 3] = 1
    return dirn

class DQN(QN):
    #TODO: ENOCODE DIRECTIONS AND PARALLELIZE!!!

    def __init__(self, env, config, logger=None):
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'Running model on device {self.device}')
        super().__init__(env, config, logger)
        self.summary_writer = SummaryWriter(self.config.output_path, max_queue=1e5)
        self.pellets = None
        
        # set multiprocessing engine
        subprocess.Popen(["ipcluster", "start", f"-n={self.config.num_cores}"])
        time.sleep(60)
        self.client = ipp.Client()
        print(f'Running on {self.client.ids} cores')
        self.view = self.client[:]
        
        # clear engines from junk
        self.view.client.purge_everything()
        
        print(f'Changing engines CWD to {os.getcwd()}')
        newdirs = [os.getcwd()]*len(self.view)
        self.view.map(os.chdir, newdirs)
        assert self.view.apply_sync(os.getcwd) == newdirs

    """
    Abstract class for Deep Q Learning
    """
    def initialize_models(self):
        """ Define the modules needed for the module to work."""
        raise NotImplementedError


    def get_q_values(self, state: Tensor, network: str) -> Tensor:
        """
        Input:
            state: A tensor of shape (batch_size, img height, img width, nchannels x config.state_history)

        Output:
            output: A tensor of shape (batch_size, num_actions)
        """
        raise NotImplementedError


    def update_target(self):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        Args:
            q_scope: name of the scope of variables for q
            target_q_scope: name of the scope of variables for the target
                network
        """
        # TODO:should I do deepcopy?
        # TODO: need .eval() at the end?
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html        
        weights = copy.deepcopy(self.q_network.state_dict())
        self.target_network.load_state_dict(weights)
        self.target_network.eval()


    def calc_loss(self, q_values : Tensor, target_q_values : Tensor, 
                    actions : Tensor, rewards: Tensor, done_mask: Tensor) -> Tensor:
        """
        Set (Q_target - Q)^2
        """
        # q_network contains states s and target network contains corresponding s'
        # line 179, 205, 211 in deep_q_learning_torch and description of sample func in replay_buffer
        raise NotImplementedError



    def add_optimizer(self) -> Optimizer:
        """
        Set training op wrt to loss for variable in scope
        """
        # TODO: I don't think I need to initialize learning rate and epsilon as these are 
        # initialized and updated in super classes
        self.optimizer = torch.optim.Adam(self.q_network.parameters())


    def process_state(self, state : Tensor) -> Tensor:
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = state.float()
#         state /= self.config.high

        return state


    def build(self):
        """
        Build model by adding all necessary variables
        """
        self.initialize_models()
        if hasattr(self.config, 'load_path'):
            print('Loading parameters from file:', self.config.load_path)
            load_path = Path(self.config.load_path)
            assert load_path.is_file(), f'Provided load_path ({load_path}) does not exist'
            self.q_network.load_state_dict(torch.load(load_path, map_location='cpu'))
            self.update_target()
            print('Load successful!')
        else:
            print('Initializing parameters randomly')
            def init_weights(m):
                if hasattr(m, 'weight'):
                    nn.init.xavier_uniform_(m.weight, gain=2 ** (1. / 2))
                if hasattr(m, 'bias'):
                    nn.init.zeros_(m.bias)
            self.q_network.apply(init_weights)
        self.q_network = self.q_network.to(self.device)
        self.target_network = self.target_network.to(self.device)
        self.add_optimizer()


    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # synchronise q and target_q networks
        assert self.q_network is not None and self.target_network is not None, \
            'WARNING: Networks not initialized. Check initialize_models'
        self.update_target()
        self.pellets = np.load('utils/pellets.npy')

       
    def add_summary(self, latest_loss, latest_total_norm, t):
        """
        Tensorboard stuff
        """
        self.summary_writer.add_scalar('loss', latest_loss, t)
        self.summary_writer.add_scalar('grad_norm', latest_total_norm, t)
        self.summary_writer.add_scalar('Avg_Reward', self.avg_reward, t)
        self.summary_writer.add_scalar('Max_Reward', self.max_reward, t)
        self.summary_writer.add_scalar('Std_Reward', self.std_reward, t)
        self.summary_writer.add_scalar('Avg_Q', self.avg_q, t)
        self.summary_writer.add_scalar('Max_Q', self.max_q, t)
        self.summary_writer.add_scalar('Std_Q', self.std_q, t)
        self.summary_writer.add_scalar('Eval_Reward', self.eval_reward, t)


    def save(self):
        """
        Saves session
        """
        # if not os.path.exists(self.config.model_output):
        #     os.makedirs(self.config.model_output)
        torch.save(self.q_network.state_dict(), self.config.model_output)
        # self.saver.save(self.sess, self.config.model_output)


    def get_best_action(self, state: Tensor) -> Tuple[int, np.ndarray]:
        """
        Return best actionrun

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        raise NotImplementedError

    def _pseudo_r_done(self, s_batch, sp_batch, r_batch, dead):
        """
        Return size (batch_size, num_channels)
        """
        debug = False
        if debug: print('debugging pseudo reward')
        
        assert self.pellets is not None
        # TODO: A BEFORE BATCH MUST BE INPUT!!!
        # TODO: fix multiple frames input
        # TODO: parallelize this
        ch_len = self.config.nb_pawns# + 1
        done = np.zeros((sp_batch.shape[0], ch_len))
        rewards = np.zeros((sp_batch.shape[0], ch_len)) + self.config.pseudo_reward_neg
        # do this for each netry in the batch
       
        var = self.view.map_sync(partial(
                parallel_func, sp_batch, r_batch, self.config.pseudo_reward_pos, self.pellets,
                              ch_len, debug, self.config.ghost_dist), range(sp_batch.shape[0]))
        
        for row in range(sp_batch.shape[0]):
            rewards[row] = var[row][0]
            done[row] = var[row][-1]
        
        return rewards, done
        
    def reduce_states(self, s_batch):
        """Returns (batch_size, H*W+nb_dirs, 1 + nb_pawns)"""
        # cut pellets
        ch_len = 1 + self.config.nb_pawns
        tot_ch_len = ch_len + 1 + 4 + 4
        s_batch = s_batch[..., :tot_ch_len]
        
        # returns (batch size, 9 nb_objects, 4 directions)
        s_dirs = np.array(self.view.map_sync(partial(get_directions, s_batch, ch_len), 
                                             range(s_batch.shape[0])))
        
        # fill in zeros for pellets
        var = np.zeros((s_dirs.shape[0], ch_len, s_dirs.shape[-1]))
        var[:, :s_dirs.shape[1], :] = s_dirs
        s_dirs = var.copy()
        
        # cut second frame
        s_batch = s_batch[..., :ch_len]
        
        # flatten states and directions
        n, h, w, c = s_batch.shape
        
        s_batch_flat = s_batch.reshape(n, h*w, c)

        # shape n, h*w+d, c
        output = np.concatenate((s_batch_flat.transpose(0,2,1), s_dirs), axis=2).transpose(0,2,1)
        
        assert output.shape == (n, h*w+4, c)

#         # shape n, h*w+d, c
#         var = np.concatenate((s_batch_flat.transpose(0,2,1), s_dirs), axis=2)
#         # add first axis
#         var_1st_r = np.repeat(var[:,0,:], repeats=var.shape[1], axis=0).reshape(-1,var.shape[1],var.shape[2])
#         output = np.concatenate((var, var_1st_r), axis=2).transpose(0,2,1)

        return output
        
    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """
        
        # set the training flag
        self.q_network.train()
        self.target_network.train()
        
        self.timer.start('update_step/replay_buffer.sample')
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch, dead_mask_batch = replay_buffer.sample(
            self.config.batch_size)
        
#        print(s_batch.shape, a_batch.shape, r_batch.shape, sp_batch.shape, done_mask_batch.shape, dead_mask_batch.shape)
        s_batch, sp_batch = s_batch//255, sp_batch//255
        r_batch, done_mask_batch = self._pseudo_r_done(s_batch, sp_batch, r_batch, dead_mask_batch)
        
        s_batch, sp_batch = self.reduce_states(s_batch), self.reduce_states(sp_batch)
        
        self.timer.end('update_step/replay_buffer.sample')
        
        assert self.q_network is not None and self.target_network is not None, \
            'WARNING: Networks not initialized. Check initialize_models'
        assert self.optimizer is not None, \
            'WARNING: Optimizer not initialized. Check add_optimizer'

        # Convert to Tensor and move to correct device
        self.timer.start('update_step/converting_tensors')
        s_batch = torch.tensor(s_batch, dtype=torch.uint8, device=self.device)
        a_batch = torch.tensor(a_batch, dtype=torch.uint8, device=self.device)
        #TODO: double check int8 and that it is cast properly later on
        r_batch = torch.tensor(r_batch, dtype=torch.int8, device=self.device)
        sp_batch = torch.tensor(sp_batch, dtype=torch.uint8, device=self.device)
        done_mask_batch = torch.tensor(done_mask_batch, dtype=torch.bool, device=self.device)
        dead_mask_batch = torch.tensor(dead_mask_batch, dtype=torch.bool, device=self.device)
        self.timer.end('update_step/converting_tensors')

        # Reset Optimizer
        self.timer.start('update_step/zero_grad')
        self.optimizer.zero_grad()
        self.timer.end('update_step/zero_grad')

        # Run a forward pass
        self.timer.start('update_step/forward_pass_q')
        s = self.process_state(s_batch)
        q_values = self.get_q_values(s, 'q_network')
        self.timer.end('update_step/forward_pass_q')

        self.timer.start('update_step/forward_pass_target')
        with torch.no_grad():
            sp = self.process_state(sp_batch)
            target_q_values = self.get_q_values(sp, 'target_network')
        self.timer.end('update_step/forward_pass_target')

        self.timer.start('update_step/loss_calc')
        loss = self.calc_loss(q_values, target_q_values, 
            a_batch, r_batch, done_mask_batch)
        self.timer.end('update_step/loss_calc')
        self.timer.start('update_step/loss_backward')
        loss.backward()
        self.timer.end('update_step/loss_backward')

        # Clip norm
        self.timer.start('update_step/grad_clip')
        total_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.clip_val)
        self.timer.end('update_step/grad_clip')

        # Update parameters with optimizer
        self.timer.start('update_step/optimizer')
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.step()
        self.timer.end('update_step/optimizer')
        
        # toggle the training flag
        self.q_network.eval()
        self.target_network.eval()
        
        return loss.item(), total_norm#.item()


    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.update_target()

