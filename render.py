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
import matplotlib.pyplot as plt


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



env = rendezvous.RendezvousEnv(nr_agents=5,
                               obs_mode='fix',
                               comm_radius=200 * np.sqrt(2),
                               world_size=200,
                               distance_bins=8,
                               bearing_bins=8,
                               torus=False,
                               dynamics='unicycle_wheel'                               )

#use state_norm when training used state_norm trick
#state_norm = Normalization(shape=env.observation_space.shape[0])

agent_net = Actor_Gaussian(max_action=1.0,hidden_width=128,state_dim=env.observation_space.shape[0],action_dim=2)
agent_net.load_state_dict(torch.load("agent.pth"))


nb_eval_steps = 300
episodes = 1
plt.ion()  # 开启交互模式
plt.subplots()


for ep in range(episodes):
    ob = env.reset()

    print(ob.shape)
    for t_rollout in range(nb_eval_steps):
        a = agent_net(ob)

        a=torch.squeeze(a, 0).detach().numpy()
        ob, r, done, info = env.step(a)
        ob=torch.tensor(ob, dtype=torch.float32)
        plt.clf()
        plt.xlim(0, 200)  # 因为清空了画布，所以要重新设置坐标轴的范围
        plt.ylim(0, 200)

        for i in range(5):
            plt.scatter(env.agents[i].state.p_pos[0], env.agents[i].state.p_pos[1])

        plt.pause(0.01)
        if done or t_rollout == nb_eval_steps - 1:
            plt.ioff()
            plt.show()

            break