import numpy as np
import matplotlib.pyplot as plt

from device_rl.data import Data
from device_rl.module import Module

class SimpleEnv():

    def __init__(self):
        self.num_envs = 4
        self.num_agents = 5

        self.observations = Data(np.zeros((self.num_envs, self.num_agents, 2)))

        actions = np.zeros((self.num_envs, self.num_agents, 2))

        # first agent in each environment moves x + 1
        for i in range(self.num_envs):
            actions[i, 0, 0] = 1
            actions[i, 0, 1] = -1

        self.actions = Data(actions)

        self._step = Module('envs/simple/step.cu').load().set_config((self.num_envs, 1), (self.num_agents, 1, 1))

        # 

        self.scs = []
        self.fig, self.axes = plt.subplots(1, 4, figsize=(20, 5), dpi=200)
        for i in range(self.num_envs):
            ax = self.axes[i]

            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.set_xticks([-10, 0, 10])
            ax.set_yticks([-10, 0, 10])

            self.scs.append(ax.scatter(
                self.observations.get()[i, :self.num_agents, 0], 
                self.observations.get()[i, :self.num_agents, 1], 
                c='orange'
            ))

        plt.ion()
        plt.tight_layout()

        self.observations.to_device()
        self.actions.to_device()
        return
    
    def reset(self):
        return
    
    def step(self):
        self._step.launch(self.observations, self.actions)
        return

    def render_grid(self):

        o = self.observations.copy_to_host().numpy()

        for i in range(self.num_envs):
            ax = self.axes[i]

            self.scs[i].set_offsets(np.column_stack((o[i, :self.num_agents, 0], o[i, :self.num_agents, 1])))
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        plt.show()