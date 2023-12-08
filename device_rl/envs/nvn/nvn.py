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
        
        self.render_config = {
            'angle_indicator_length': 100,
            'entity_linewidth': 2,
            'entity_size': 250,
            'ball_size': 100
        }

        self.NAME = 'nvn'

        self.num_envs = num_envs
        self.num_agents = num_agents
        self.num_opponents = num_opponents

        # (x, y, angle, kick)
        self.act_size = 4
        self.act = Data(np.zeros((self.num_envs, self.num_agents, self.act_size)))

        self.state = Data(np.zeros((self.num_envs, self.num_agents + self.num_opponents + 1, 4)))


        # (x, y, sin(angle), cos(angle)) for ball, goal, and opp. goal
        # + (can_kick)
        # + (x, y, sin(angle), cos(angle)) for all other agents and opponents
        self.obs_size = (4 * 3) + 1 + (4 * (self.num_agents - 1 + self.num_opponents))
        self.obs = Data(np.zeros((self.num_envs, self.num_agents, self.obs_size)))

        self.rew = Data(np.zeros((self.num_envs, self.num_agents)))
        self.term = Data(np.full((self.num_envs, self.num_agents), False))
        self.trun = Data(np.full((self.num_envs, self.num_agents), False))
        self.info = Data(np.zeros((self.num_envs, self.num_agents)))

        # (x, y, angle, velocity)
        # self.agents = Data(np.zeros((self.num_envs, self.num_agents, 4)))
        # self.opponents = Data(np.zeros((self.num_envs, self.num_opponents, 4)))
        # self.ball = Data(np.zeros((self.num_envs, 1, 4)))

        # reward_map
        self.rew_map = Data(np.array([
            10000,  # goal
            0.1,    # ball to goal
            10,     # kick
            -1,     # missed_kick
            -10,    # contact
            -10,    # out of bounds
        ]))

        self.goal_scored = Data(np.zeros((self.num_envs, 1), dtype=np.int32))

        self.width = int(np.ceil(np.sqrt(self.num_envs)))
        self.height = int(np.floor(np.sqrt(self.num_envs)))

        n_a = self.num_agents
        n_o = self.num_opponents

        self.scs = []
        self.lines = []
        self.angle_indicators = []
        self.fig, self.axes = plt.subplots(self.height, self.width, figsize=(24, 20), dpi = 200)
        self.axes = self.axes.flatten()

        state = self.state.get()
        for i in range(self.num_envs):

            ax = self.axes[i]

            ax.set_xlim(-5000, 5000)
            ax.set_ylim(-3500, 3500)
            ax.set_xticks([])
            ax.set_yticks([])

            circle = plt.Circle([0, 0], 1000, edgecolor='white', facecolor='none', alpha = 0.4) 
            ax.add_patch(circle)

            field = plt.Rectangle((-4500, -3000), 9000, 6000, edgecolor='white', facecolor='none', alpha = 0.4)
            ax.add_patch(field)

            goal_left = plt.Rectangle((-4500, -750), 750, 1500, edgecolor='white', facecolor='none', alpha = 0.4)
            ax.add_patch(goal_left)

            goal_right = plt.Rectangle((4500 - 750, -750), 750, 1500, edgecolor='white', facecolor='none', alpha = 0.4)
            ax.add_patch(goal_right)

            ax.plot([0, 0], [-3000, 3000], color='white', linestyle='-', alpha = 0.4)


            lines = []
            for j in range(self.num_agents + self.num_opponents + 1):
                lines.append(
                    ax.plot(
                        [state[i, j, 0], state[i, j, 0] + self.render_config['angle_indicator_length'] * np.cos(state[i, j, 2])],
                        [state[i, j, 1], state[i, j, 1] + self.render_config['angle_indicator_length'] * np.sin(state[i, j, 2])],
                        color = 'white'
                    )[0]
                )
            
            self.angle_indicators.append(lines)

            self.scs.append((
                ax.scatter(
                    state[i, :n_a, 0], 
                    state[i, :n_a, 1], 
                    edgecolor = 'blue',
                    marker = 'o',
                    facecolor = 'none',
                    linewidths = self.render_config['entity_linewidth'],
                    s = self.render_config['entity_size']
                ),
                ax.scatter(
                    state[i, n_a:(n_a + n_o), 0], 
                    state[i, n_a:(n_a + n_o), 1], 
                    edgecolor = 'red',
                    marker = 'o',
                    facecolor = 'none',
                    linewidths = self.render_config['entity_linewidth'],
                    s = self.render_config['entity_size']
                ),
                ax.scatter(
                    state[i, -1, 0], 
                    state[i, -1, 1], 
                    c = 'white',
                    marker = '.',
                    s = self.render_config['ball_size']
                )
            ))

        plt.ion()
        plt.tight_layout()

        self.module = Module(f'envs/{self.NAME}/{self.NAME}.cu')
        self.module.load()
        self.module.set_config((self.num_envs, 1), (self.num_agents + self.num_opponents + 1, 1, 1))

        self.num_agents = Data(self.num_agents)
        self.num_opponents = Data(self.num_opponents)

        self.num_agents.to_device()
        self.num_opponents.to_device()
        self.obs.to_device()
        self.act.to_device()
        self.rew.to_device()
        self.term.to_device()
        self.trun.to_device()
        self.info.to_device()
        self.state.to_device()
        self.rew_map.to_device()
        self.goal_scored.to_device()
            
    def reset(self):
        self.module.launch('reset')(
            self.obs,
            self.state,
            self.num_agents,
            self.num_opponents
            # self.agents,
            # self.opponents,
            # self.ball,
            # self.goal_scored,
        )
        return
    
    def test(self):
        self.module.launch('test')(

        )

    def sample(self):
        self.module.launch('sample')(
            self.act
        )
    
    def step(self):
        self.module.launch('step')(
            self.state,
            self.act,
            self.num_agents,
            self.num_opponents,
            self.obs,
            self.rew,
            self.term,
            self.trun,
            self.info
        )
        return

    def render(self, grid = True):
        state = self.state.copy_to_host().numpy()

        n_a = self.num_agents.get()
        n_o = self.num_opponents.get()


        for i in range(self.num_envs):
            ax = self.axes[i]

            for j, line in enumerate(self.angle_indicators[i]):
                line.set_data(
                    [state[i, j, 0], state[i, j, 0] + self.render_config['angle_indicator_length'] * np.cos(state[i, j, 2])],
                    [state[i, j, 1], state[i, j, 1] + self.render_config['angle_indicator_length'] * np.sin(state[i, j, 2])],
                )

            self.scs[i][0].set_offsets(
                np.column_stack((
                    state[i, :n_a, 0], 
                    state[i, :n_a, 1]
                ))
            )

            self.scs[i][1].set_offsets(
                np.column_stack((
                    state[i, n_a:(n_a + n_o), 0], 
                    state[i, n_a:(n_a + n_o), 1]
                ))
            )

            self.scs[i][2].set_offsets(
                np.column_stack((
                    state[i, -1, 0], 
                    state[i, -1, 1]
                ))
            )

            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        plt.show()