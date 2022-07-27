# -*- coding: utf-8 -*-
"""
Created on Fri May 27 00:16:34 2022

@author: yx
"""

import torch
from ma_envs.envs.point_envs import rendezvous

import numpy as np

# from mlppolicy3 import MLPActorCritic
#from ppo_model import MLPActorCritic

import matplotlib.pyplot as plt

env = rendezvous.RendezvousEnv(nr_agents=5,
                               obs_mode='fix',
                               comm_radius=200 * np.sqrt(2),
                               world_size=100,
                               distance_bins=8,
                               bearing_bins=8,
                               torus=False,
                               dynamics='unicycle_wheel')

# agent = MLPActorCritic(env.observation_space.shape[0], env.action_space.shape[0], 256, True)
# agent = MLPActorCritic(env.observation_space, env.nr_agents, env.action_space.shape[0],128, env.observation_space.dim_local_o)


# agent.load_state_dict(torch.load("models/500000_5_fix_acc_0.0001_false_agent.pth"))

# Evaluate.


