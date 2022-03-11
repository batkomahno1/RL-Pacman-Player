import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.tensor import Tensor

import logging
import os
from typing import Tuple

from core.deep_q_learning_torch import DQN

os.environ['KMP_DUPLICATE_LIB_OK']='True'
logging.getLogger('matplotlib.font_manager').disabled = True

class HydraNet(nn.Module):
    def __init__(self, layers, config, training=False, nb_actions = 9):
        super(HydraNet, self).__init__()
        self.layers = layers#nn.ModuleList(layers)
        self.count = 0
        self.nb_actions = nb_actions
        self.config = config
        self.training = training
        self.head_weights = Tensor(self.config.weights).to('cuda:0')#[:,None]
    
    def forward(self, x):
        """
        input: batch_size, h*w+d, 1+nb_pawns
        output: batch_size, heads, actions
        """
        n, _, _ = x.shape
        c = self.config.nb_pawns

        # add div head channel
        if self.config.diversification: c += 1

        var = torch.zeros(n, c, self.nb_actions).to('cuda:0')

        range_ = range(c) if not self.config.diversification else range(c-1)
        
        for i in range_:
            data = x[:, :, [0, i+1]].permute(0,2,1)
#             print(data.shape)
            var[:, i, :] = self.layers(data)
                    
        # add diversification head during first n steps
        if self.training and self.count < self.config.div_steps and self.config.diversification: 
            var[:, -1, :] = torch.FloatTensor(var[:, -1, :].shape).uniform_(
                0, self.config.max_div_q_val).to('cuda:0')

        self.count += 1
        return var


class Hydra(DQN):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Model configuration can be found in the Methods section of the above paper.
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
            3. If you use OrderedDict, make sure the keys for the the layers are:
                - "0", "2", "4" for three Conv2d layers
                - "7" for the first Linear layer
                - "9" for the final Linear layer
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        num_channels = (1 + self.config.nb_pawns)
        # locations of pacman and pawns + directios of pacman and ghosts
        
        lin_input_size = 2*(img_height*img_width + 4)#, num_channels
        lin_output_size = self.config.lin_output_size
        
        # layers
        var = [nn.Flatten(), 
               nn.Linear(lin_input_size, lin_output_size//2), nn.ReLU(), 
               nn.Linear(lin_output_size//2, lin_output_size//4), nn.ReLU(), 
               nn.Linear(lin_output_size//4, lin_output_size), nn.ReLU(),
               nn.Linear(lin_output_size, num_actions)]

        layers = nn.Sequential(*var)
        self.q_network = HydraNet(layers, self.config, training=False, 
                                  nb_actions = num_actions).to(self.device)
        
        self.target_network = HydraNet(layers, self.config, training=False, 
                                       nb_actions = num_actions).to(self.device)

    def calc_loss(self, q_values : Tensor, target_q_values : Tensor,
                    actions : Tensor, rewards: Tensor, done_mask: Tensor) -> Tensor:
        """
        Calculate the MSE loss of this step.
        The loss for an example is defined as:
            Q_samp(s) = r if done
                        = r + gamma * max_a' Q_target(s', a') otherwise
            loss = (Q_samp(s) - Q(s, a))^2

        Args:
            q_values: (torch tensor) shape = (batch_size, nb_pawns, num_actions)
                The Q-values that your current network estimates (i.e. Q(s, a') for all a')
            target_q_values: (torch tensor) shape = (batch_size, nb_pawns, num_actions)
                The Target Q-values that your target network estimates (i.e. (i.e. Q_target(s', a') for all a')
            actions: (torch tensor) shape = (batch_size,)
                The actions that you actually took at each step (i.e. a)
            rewards: (torch tensor) shape = (batch_size, nb_pawns)
                The rewards that you actually got at each step (i.e. r)
            done_mask: (torch tensor) shape = (batch_size, nb_pawns)
                A boolean mask of examples where we reached the terminal state

            You can treat `done_mask` as a 0 and 1 where 0 is not done and 1 is done using torch.type as
            done below

            To extract Q(a) for a specific "a" you can use the torch.sum and torch.nn.functional.one_hot. 
            Think about how.
        """
        # you may need this variable
        gamma = self.config.gamma
        done_mask = done_mask.type(torch.int)
        actions = actions.type(torch.int64)
        
        # q_network contains states s and target network contains corresponding s'
        # line 179, 205, 211 in deep_q_learning_torch and description of sample func in replay_buffer
        
        y = (rewards+gamma*torch.max(target_q_values, -1)[0])+rewards # batch x heads
        a = actions[:, None].repeat(1, q_values.shape[1])
        
        x = torch.sum(q_values*F.one_hot(a, q_values.shape[-1]), -1) # batch x heads

        if self.config.diversification: 
            w = torch.Tensor(np.append(self.config.weights, 1)).to('cuda:0') # (nb_pawns, )
        else:
            w = torch.Tensor(self.config.weights).to('cuda:0') # (nb_pawns, )

        var = torch.sum(((y-x)@w[:,None])**2, 1)# batch 
        
#         var = torch.sum((y - x)**2, 1)# batch # (heads, )
        
        loss = torch.sum(var, -1)/var.shape[0]
        
        return loss

    def get_best_action(self, state: Tensor) -> Tuple[int, np.ndarray]:
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        state = self.reduce_states(state if len(state.shape) == 4 else state[np.newaxis, :])
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.uint8, device=self.device)#.unsqueeze(0)
            s = self.process_state(s)
            #q_values: (torch tensor) shape = (batch_size, nb_pawns, num_actions)
            q_vals = self.get_q_values(s, 'q_network') #(batch_size, nb_pawns, num_actions)
            
            if self.config.diversification: 
                w = torch.Tensor(np.append(self.config.weights, 1)).to('cuda:0') # (nb_pawns, )
            else:
                w = torch.Tensor(self.config.weights).to('cuda:0') # (nb_pawns, )
            
            var = q_vals.permute(0, 2, 1)@w #(batch_size, num_actions)

            action_values = var.to('cpu').tolist() 

        action = np.argmax(action_values)
        return action, action_values

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None        
        # the input should be (batch_size, in_channels, H, W) but we have(batch_size, H, W, in_channels)
        # https://engineering.purdue.edu/DeepLearn/pdf-kak/week6.pdf                
        # swap axis
        if network=='q_network':
            out = self.q_network(state)
        else:
            out = self.target_network(state)
        return out