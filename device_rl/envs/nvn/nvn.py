import numpy as np
import matplotlib.pyplot as plt

from device_rl.data import Data
from device_rl.module import Module

class NvNEnv():

    def __init__(
        self, 
        num_envs = 4, 
        num_agents = 5, 
        num_opponents = 5
    ):
        self.NAME = 'nvn'

        self.num_envs = num_envs
        self.num_agents = num_agents
        self.num_opponents = num_opponents

        # (x, y, sin(angle), cos(angle)) for ball, goal, and opp. goal
        # + (can_kick)
        # + (x, y, sin(angle), cos(angle)) for all other agents and opponents
        self.obs_size = (4 * 3) + 1 + (4 * (self.num_agents - 1 + self.num_opponents))
        self.observations = Data(np.zeros((self.num_envs, self.num_agents, self.obs_size)))

        # (x, y, angle, kick)
        self.act_size = 4
        self.actions = Data(np.zeros((self.num_envs, self.num_agents, self.act_size)))

        # (x, y, angle, velocity)
        self.agents = Data(np.zeros((self.num_envs, self.num_agents, 4)))
        self.opponents = Data(np.zeros((self.num_envs, self.num_opponents, 4)))
        self.ball = Data(np.zeros((self.num_envs, 1, 4)))

        # reward_map
        self.reward_map = Data(np.array([
            10000,  # goal
            0.1,    # ball to goal
            10,     # kick
            -1,     # missed_kick
            -10,    # contact
            -10,    # out of bounds
        ]))

        self.goal_scored = Data(np.zeros((self.num_envs, 1), dtype=np.int32))

        self.observations.to_device()
        self.actions.to_device()
        self.agents.to_device()
        self.opponents.to_device()
        self.ball.to_device()
        self.reward_map.to_device()
        self.goal_scored.to_device()

        self.module = Module(f'envs/{self.NAME}/{self.NAME}.cu')
        self.module.load()
        self.module.set_config((self.num_envs, 1), (self.num_agents, 1, 1))
            
    def reset(self):
        self.module.launch('reset')(
            self.observations,
            self.agents,
            self.opponents,
            self.ball,
            self.goal_scored,
        )
        return
    
    def step(self):
        # self.module.launch('step')(
        #     self.observations, 
        #     self.actions
        # )
        return

    def render(grid = True):
        return