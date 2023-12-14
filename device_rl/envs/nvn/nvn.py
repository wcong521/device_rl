import time

import numpy as np
import matplotlib.pyplot as plt
from rich import print

from device_rl.data import Data
from device_rl.module import Module

class NvNEnv():

    def __init__(
        self, 
        n_envs = 4, 
        n_agents = 5, 
        n_opponents = 5,
        log_time = False,
        render = True,
    ):

        self.NAME = 'nvn'

        # stores the internal state of the agent, opponent, and ball
        # (x, y, angle, x_vel, y_vel, angle_vel, kick) for agent and opponent
        # (x, y, angle, vel, x_prev, y_prev) for ball
        # must be the maximum length
        self.STATE_SIZE = 7

        # stores the action for each agent and opponent
        # (x, y, angle, kick)
        self.ACTION_SIZE = 4

        # stores the observation for each agent
        # (x, y, sin(angle), cos(angle)) for ball, goal, and opp. goal
        # (can_kick)
        # (x, y, sin(angle), cos(angle)) for all other agents and opponents
        self.OBS_SIZE = (4 * 3) + 1 + (4 * (n_agents - 1 + n_opponents))

        self._render = render

        # GPU variables

        self.action = Data(np.zeros((n_envs, n_agents + n_opponents, self.ACTION_SIZE)))
        self.action_size = Data(self.ACTION_SIZE)

        self.state = Data(np.zeros((n_envs, n_agents + n_opponents + 1, self.STATE_SIZE)))
        self.state_size = Data(self.STATE_SIZE)

        self.obs = Data(np.zeros((n_envs, n_agents + n_opponents, self.OBS_SIZE)))
        self.obs_size = Data(self.OBS_SIZE)

        self.rew = Data(np.zeros((n_envs, n_agents)))
        self.term = Data(np.full((n_envs, n_agents), False))
        self.trun = Data(np.full((n_envs, n_agents), False))
        self.info = Data(np.zeros((n_envs, n_agents)))

        self.rew_map = Data(np.array([
            10000,  # goal
            0.1,    # ball to goal
            10,     # kick
            -1,     # missed_kick
            -10,    # contact
            -10,    # out of bounds
        ]))

        self.goal_scored = Data(np.zeros((n_envs, 1), dtype=np.int32))

        self.n_agents = Data(n_agents)
        self.n_opponents = Data(n_opponents)
        self.n_envs = Data(n_envs)

        self.time = Data(np.array([[0]], dtype=np.int32));



        if (self._render):

            # visualization
            # must be initialized before GPU variables are sent to GPU
            start = time.time()

            self.render_config = {
                'angle_indicator_length': 200,
                'entity_linewidth': 2,
                'entity_size': 250,
                'ball_size': 150,
                'field_opacity': 0.3
            }

            self.render_init(n_envs, n_agents, n_opponents)

            end = time.time()
            if (log_time): print(f"Initialized visualization in [orange1]{end - start}[/orange1]s.")



        # send to GPU (device)

        start = time.time()

        self.state_size.to_device()
        self.action_size.to_device()
        self.n_agents.to_device()
        self.n_opponents.to_device()
        self.obs.to_device()
        self.action.to_device()
        self.rew.to_device()
        self.term.to_device()
        self.trun.to_device()
        self.info.to_device()
        self.state.to_device()
        self.rew_map.to_device()
        self.goal_scored.to_device()
        self.time.to_device()

        end = time.time()
        if (log_time): print(f"Host to device transfer in [orange1]{end - start}[/orange1]s.")


        # kernel configuration

        # 1 block = 1 env
        self.grid_dim = (n_envs, 1)

        # 1 thread for each agent, opponent, and ball
        self.block_dim = (n_agents + n_opponents + 1, 1, 1)

        # size of shared memory
        # all state and action vectors corresponding to a single environment
        # obs is not loaded into shared memory because only a single write operation is performed
        self.shared_size = (n_agents + n_opponents + 1) * self.STATE_SIZE * self.state.get().element_size() + \
                           (n_agents + n_opponents) * self.ACTION_SIZE * self.action.get().element_size()

        # load the .cu file
        start = time.time()
        
        self.module = Module(f'envs/{self.NAME}/{self.NAME}.cu')
        self.module.load()

        end = time.time()
        if (log_time): print(f"Loaded module in [orange1]{end - start}[/orange1]s.")

    def render_init(self, n_envs, n_agents, n_opponents):
        width = int(np.ceil(np.sqrt(n_envs)))
        height = int(np.floor(np.sqrt(n_envs)))

        self.scs = []
        self.lines = []
        self.angle_indicators = []
        self.annotations = []
        self.fig, self.axes = plt.subplots(height, width, figsize = (28, 20), dpi = 200)
        self.axes = self.axes.flatten()

        state = self.state.get()
        for i in range(n_envs):

            ax = self.axes[i]

            ax.invert_yaxis()
            ax.set_xlim(-5200, 5200)
            ax.set_ylim(-3700, 3700)
            ax.set_xticks([])
            ax.set_yticks([])

            circle = plt.Circle(
                [0, 0], 1000, 
                edgecolor='white', 
                facecolor='none', 
                alpha = self.render_config['field_opacity']
            ) 
            ax.add_patch(circle)

            field = plt.Rectangle(
                (-4500, -3000), 9000, 6000, 
                edgecolor='white', 
                facecolor='none', 
                alpha = self.render_config['field_opacity']
            )
            ax.add_patch(field)

            goal_left = plt.Rectangle(
                (-4500, -750), 750, 1500, 
                edgecolor = 'white', 
                facecolor = 'none', 
                alpha = self.render_config['field_opacity']
            )
            ax.add_patch(goal_left)

            goal_right = plt.Rectangle(
                (4500 - 750, -750), 750, 1500, 
                edgecolor = 'white', 
                facecolor = 'none', 
                alpha = self.render_config['field_opacity']
            )
            ax.add_patch(goal_right)

            ax.plot(
                [0, 0], [-3000, 3000], 
                color = 'white', 
                linestyle = '-', 
                alpha = self.render_config['field_opacity']
            )

            lines = []
            for j in range(n_agents + n_opponents + 1):
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
                    state[i, :n_agents, 0], 
                    state[i, :n_agents, 1], 
                    edgecolor = 'blue',
                    marker = 'o',
                    facecolor = 'none',
                    linewidths = self.render_config['entity_linewidth'],
                    s = self.render_config['entity_size']
                ),
                ax.scatter(
                    state[i, n_agents:(n_agents + n_opponents), 0], 
                    state[i, n_agents:(n_agents + n_opponents), 1], 
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

            # self.annotations.append(
            #     [ax.annotate(
            #         str(j),
            #         xy = (state[i, j, 0], state[i, j, 1] + 500),
            #         ha='center', va='center', fontsize=10
            #     ) for j in range(n_agents + n_opponents + 1)]
            # )

        plt.ion()
        plt.tight_layout()
            
    def reset(self):
        self.module.launch(
            'reset', 
            grid = self.grid_dim, 
            block = self.block_dim,
        )(
            self.state,
            self.state_size,
            self.goal_scored,
            self.n_agents,
            self.n_opponents,
            # output
            self.obs,
            Data(np.random.randint(1, 1000000))
        )
        return
    
    def test(self):
        self.module.launch(
            'test',
            grid = self.grid_dim, 
            block = self.block_dim,
        )()

    def sample(self):
        self.module.launch(
            'sample',
            grid = self.grid_dim, 
            block = self.block_dim,
        )(
            self.action,
            self.action_size,
            Data(np.random.randint(1, 1000000))
        )
    
    def step(self):
        self.module.launch(
            'step',
            grid = self.grid_dim, 
            block = self.block_dim,
            shared = self.shared_size
        )(
            # inputs
            self.state,
            self.state_size,
            self.action,
            self.action_size,
            self.obs_size,
            self.rew_map,
            self.goal_scored,
            self.n_agents,
            self.n_opponents,
            self.time,
            Data(np.random.randint(1, 1000000)),

            # outputs
            self.obs,
            self.rew,
            self.term,
            self.trun,
            self.info,
        )
        return

    def render(self, grid = True):
        if not self._render:
            return


        state = self.state.copy_to_host().numpy()

        n_agents = self.n_agents.get()
        n_opponents = self.n_opponents.get()


        for i in range(self.n_envs.get()):
            ax = self.axes[i]

            for j, line in enumerate(self.angle_indicators[i]):
                line.set_data(
                    [state[i, j, 0], state[i, j, 0] + self.render_config['angle_indicator_length'] * np.cos(state[i, j, 2])],
                    [state[i, j, 1], state[i, j, 1] + self.render_config['angle_indicator_length'] * np.sin(state[i, j, 2])],
                )

            # for j in range(n_agents + n_opponents + 1):
            #     self.annotations[i][j].set_position((
            #         state[i, j, 0], state[i, j, 1] + 500
            #     ))


            self.scs[i][0].set_offsets(
                np.column_stack((
                    state[i, :n_agents, 0], 
                    state[i, :n_agents, 1]
                ))
            )

            self.scs[i][1].set_offsets(
                np.column_stack((
                    state[i, n_agents:(n_agents + n_opponents), 0], 
                    state[i, n_agents:(n_agents + n_opponents), 1]
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