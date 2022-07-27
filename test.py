# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:21:52 2022

@author: yx
"""

from ma_envs.envs.point_envs import rendezvous
import torch
import numpy as np
from normalization import Normalization
import torch.nn as nn

class Actor_Gaussian(nn.Module):
    def __init__(self, max_action,hidden_width,state_dim,action_dim):
        super(Actor_Gaussian, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # We use 'nn.Paremeter' to train log_std automatically
        self.activate_func = nn.Tanh()  # Trick10: use tanh


    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean





def evaluate_policy( env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent(torch.tensor(s,dtype=torch.float))  # We use the deterministic policy during the evaluating
            action = torch.squeeze(a,0).detach().numpy()
            s_, r, done, _ = env.step(action)
            s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


env = rendezvous.RendezvousEnv(nr_agents=5,
                               obs_mode='fix',
                               comm_radius=200 * np.sqrt(2),
                               world_size=200,
                               distance_bins=8,
                               bearing_bins=8,
                               torus=False,
                               dynamics='unicycle_wheel'                               )


state_norm = Normalization(shape=env.observation_space.shape[0])
agent_net = Actor_Gaussian(max_action=1.0,hidden_width=128,state_dim=env.observation_space.shape[0],action_dim=2)
agent_net.load_state_dict(torch.load("agent.pth"))


# print(evaluate_policy(env,agent_net,state_norm))


def test_dm(env2):
    s = env2.reset2()
    for i in range(5):
        print(env2.world.agents[i].state.p_pos)
        print(env2.world.agents[i].state.p_orientation)
    print(env2.world.distance_matrix)
    print(env2.world.angle_matrix)
    print(env2.world.angles_shift)
    print(env2.world.angles)




test_dm(env)