import copy
import gymnasium as gym
import numpy as np
import sys

from device_rl.abstract_sim.base import BaseEnv
from supersuit import clip_actions_v0

from device_rl.abstract_sim.base import BaseEnv


def env(render_mode=None, opponents=None, opp_deterministic=False, prob_psudo_random=0):
    env = parallel_env(
        opponents=opponents,
        opp_deterministic=opp_deterministic,
        prob_psudo_random=prob_psudo_random,
    )
    return env


class parallel_env(BaseEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        opponents=None,
        opp_deterministic=False,
        prob_psudo_random=0,
        render_mode="rgb_array",
    ):
        # Init base class
        super().__init__()
        
        device = "cpu"

        """
        Required:
        - possible_agents
        - action_spaces
        - observation_spaces
        """
        self.random = False
        if opponents:
            # Load opponent policies. List of paths. Want to load policies and names
            self.opponent_policies = []
            for opp in opponents:
                # Extract number from opp. Format is ./policies/env_policy_0.zip
                # Check if opp.split('_')[-1].split('.')[0] is an int
                if opp.split("_")[-1].split(".")[0].isdigit():
                    opp_num = int(opp.split("_")[-1].split(".")[0])
                    self.opponent_policies.append([opp_num, PPO.load(opp, device=device)])
                else:
                    self.opponent_policies.append([0, PPO.load(opp, device=device)])
        else:
            self.random = True

        self.opp_deterministic = opp_deterministic
        self.prob_psudo_random = prob_psudo_random

        # agents
        self.possible_agents = [f'agent_{i}' for i in range(15)]
        # self.possible_agents = ["agent_0"]
        self.agents = self.possible_agents[:]
        self.agent_idx = {agent: i for i, agent in enumerate(self.agents)}

        self.possible_opponents = [f'opp_{i}' for i in range(15)]
        # self.possible_opponents = ["opp_0"]
        self.opponents = self.possible_opponents[:]
        self.opponent_idx = {opponent: i for i, opponent in enumerate(self.opponents)}

        # 3D vector of (x, y, angle, kick) velocity changes
        self.action_spaces = {
            agent: gym.spaces.Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]))
            for agent in self.agents
        }

        # observation spaces
        obs_space_components = [
            "ball pos",
            "goal pos",
            "opp goal",
        ]  # Plus 1 hot for can kick and 4 for each other agent
        obs_size = (
            1
            + 4 * len(obs_space_components)
            + (len(self.possible_agents) + len(self.possible_opponents) - 1) * 4
        )
        self.observation_spaces = {
            agent: gym.spaces.Box(low=-1, high=1, shape=(obs_size,))
            for agent in self.agents
        }

        self.episode_length = 4000

        self.ball_acceleration = -0.8
        self.ball_velocity_coef = 3
        self.last_touched = None

        self.displacement_coef = 0.06
        self.angle_displacement = 0.05
        self.robot_radius = 20

        self.x_disp = 1.5
        self.y_disp = 1.5
        self.angle_disp = 1.2

        self.max_velocities = {"x": 2, "y": 2, "angle": 2}

        self.reward_dict = {
            "goal": 10000,  # Team
            "goal_scored": False,
            "ball_to_goal": 0.1,  # Team
            "kick": 10,  # Individual
            "missed_kick": -1,  # Individual
            "contact": -10,  # Individual
            "out_of_bounds": -10,  # Individual
        }

    def get_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1[:2]) - np.array(pos2[:2]))

    def can_kick(self, agent, opp=False):
        if not opp:
            i = self.agent_idx[agent]
            agent_loc = self.robots[i]
        else:
            i = self.opponent_idx[agent]
            agent_loc = self.opp_robots[i]

        if self.check_facing_ball(agent, opp=opp):
            return (
                self.get_distance(agent_loc, self.ball)
                < (self.robot_radius + self.ball_radius) * 10
            )

    """
    ego-centric observation:
        origin,
        goal,
        other robots,
        ball
    """

    def get_obs(self, agent):
        i = self.agent_idx[agent]
        agent_loc = self.robots[i]

        obs = []

        # Ball
        ball = self.get_relative_observation(agent_loc, self.ball)
        obs.extend(ball)

        # 1 hot for can kick
        obs.extend([1] if self.can_kick(agent) else [-1])

        # Teammates
        for teammates in self.agents:
            j = self.agent_idx[teammates]
            if i != j:
                teammate_loc = self.robots[self.agent_idx[teammates]]
                obs.extend(self.get_relative_observation(agent_loc, teammate_loc))

        # Opponents
        for opp in self.opponents:
            opp_loc = self.opp_robots[self.opponent_idx[opp]]
            obs.extend(self.get_relative_observation(agent_loc, opp_loc))

        # Goal
        goal = self.get_relative_observation(agent_loc, [4800, 0])
        obs.extend(goal)

        # Opponent goal
        opp_goal = self.get_relative_observation(agent_loc, [-4800, 0])
        obs.extend(opp_goal)

        return np.array(obs, dtype=np.float32)

    def opp_get_obs(self, agent):
        i = self.opponent_idx[agent]
        agent_loc = self.opp_robots[i]

        obs = []

        # Ball
        ball = self.get_relative_observation(agent_loc, self.ball)
        obs.extend(ball)

        # 1 hot for can kick
        obs.extend([1] if self.can_kick(agent, opp=True) else [-1])

        # Opponent
        for opp in self.opponents:
            if opp != agent:
                opp_loc = self.opp_robots[self.opponent_idx[opp]]
                obs.extend(self.get_relative_observation(agent_loc, opp_loc))

        # Teammates
        for teammates in self.agents:
            teammate_loc = self.robots[self.agent_idx[teammates]]
            obs.extend(self.get_relative_observation(agent_loc, teammate_loc))

        # Goal
        goal = self.get_relative_observation(agent_loc, [-4800, 0])
        obs.extend(goal)

        # Opponent goal
        opp_goal = self.get_relative_observation(agent_loc, [4800, 0])
        obs.extend(opp_goal)

        return np.array(obs, dtype=np.float32)

    """
    Format for robots:
    x, y, angle

    Format for robot velocities:
    x, y, angle

    Format for ball:
    x, y, velocity, angle
    """

    def reset(self, seed=None, return_info=False, options=None, **kwargs):
        self.time = 0

        # self.robots = [[np.random.uniform(-4500, 4500), np.random.uniform(-3000, 3000), np.random.uniform(-np.pi, np.pi)] for _ in range(len(self.agents))]
        # self.opp_robots = [[np.random.uniform(-4500, 4500), np.random.uniform(-3000, 3000), np.random.uniform(-np.pi, np.pi)] for _ in range(len(self.opponents))]

        # Spawn one robot in goal area, one robot in center
        self.robots = [
            [
                np.random.uniform(-4500, -4000),
                np.random.uniform(-1500, 1500),
                np.random.uniform(-np.pi, np.pi),
            ] for _ in range(len(self.agents))
        ]

        self.opp_robots = [
            [
                np.random.uniform(4000, 4500),
                np.random.uniform(-1500, 1500),
                np.random.uniform(-np.pi, np.pi),
            ] for _ in range(len(self.opponents))
        ]

        self.robot_velocities = [[0, 0, 0] for _ in range(len(self.agents))]
        self.opp_velocities = [[0, 0, 0] for _ in range(len(self.opponents))]

        self.reward_dict["goal_scored"] = False
        # self.previous_distances = [None for _ in range(len(self.agents))]

        # x, y, velocity, angle
        self.ball = [
            np.random.uniform(-2500, 2500),
            np.random.uniform(-1500, 1500),
            0,
            0,
        ]

        observations = {}
        for agent in self.agents:
            observations[agent] = self.get_obs(agent)

        infos = {agent: {} for agent in self.agents}

        self.psudo_random = False

        # Choose opponent policy
        if not self.random:
            # With 30% probability, choose random opponent
            if np.random.uniform(0, 1) < self.prob_psudo_random:
                self.psudo_random = True
            else:
                self.opponent_policy = self.opponent_policies[
                    np.random.randint(0, len(self.opponent_policies))
                ][1]

        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        self.time += 1

        # Copy previous locations (deep copy)
        # self.prev_locations = [loc[:] for loc in self.robots]
        self.prev_ball = copy.deepcopy(self.ball)

        # Update agent locations and ball
        for agent in self.agents:
            action = actions[agent]
            self.move_agent(agent, action)

        # Update opponent locations
        for opp in self.opponents:
            if not self.random and not self.psudo_random:
                # opp_action, _ = self.opponent_policy.predict(self.opp_get_obs(opp), deterministic=self.opp_deterministic)
                opp_action, _ = self.opponent_policy.predict(
                    self.opp_get_obs(opp), deterministic=False
                )
                self.move_opponent(opp, opp_action)
            else:
                self.move_opponent(
                    opp,
                    [
                        np.random.uniform(-1, 1),
                        np.random.uniform(-1, 1),
                        np.random.uniform(-1, 1),
                        np.random.uniform(-1, 1),
                    ],
                )

        self.update_ball()

        # Calculate rewards
        for agent in self.agents:
            obs[agent] = self.get_obs(agent)
            rew[agent] = self.calculate_reward(agent, actions[agent])
            terminated[agent] = self.time > self.episode_length
            truncated[agent] = False
            info[agent] = {}

        if self.reward_dict["goal_scored"]:
            terminated = {agent: True for agent in self.agents}
            # # Reset ball
            # self.ball = [0, 0, 0, 0]
            # self.reward_dict["goal_scored"] = False

        return obs, rew, terminated, truncated, info

    """
    Checks if ball is in goal area
    """

    def goal(self):
        if self.ball[0] > 4500 and self.ball[1] < 1100 and self.ball[1] > -1100:
            return True
        return False

    def opp_goal(self):
        if self.ball[0] < -4500 and self.ball[1] < 1100 and self.ball[1] > -1100:
            return True
        return False

    def contacting_robot(self, agent):
        # Try deepmind method?
        """
        A penalty, equal to the cosine of the angle between the players
        velocity vector and the heading of the opponent, if the player
        is within 1 m of the opponent. This discourages the agents from
        interfering with and fouling the opponent.
        """
        i = self.agent_idx[agent]
        agent_loc = self.robots[i]

        for age in self.agents:
            age_loc = self.robots[self.agent_idx[age]]
            if age != agent:
                if (
                    self.get_distance(agent_loc, age_loc)
                    < (self.robot_radius + self.robot_radius) * 8
                ):
                    return True

        for opp in self.opponents:
            opp_loc = self.opp_robots[self.opponent_idx[opp]]
            if (
                self.get_distance(agent_loc, opp_loc)
                < (self.robot_radius + self.robot_radius) * 8
            ):
                return True

        return False

    def out_of_bounds(self, agent):
        i = self.agent_idx[agent]
        agent_loc = self.robots[i]

        if (
            agent_loc[0] > 4800
            or agent_loc[0] < -4800
            or agent_loc[1] > 3200
            or agent_loc[1] < -3200
        ):
            return True
        return False

    """
    Get angle for contacting other robot
    """

    def get_angle(self, pos1, pos2):
        return np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])

    def calculate_reward(self, agent, action):
        i = self.agent_idx[agent]
        reward = 0

        info_dict = {}

        # Goal - Team
        if self.goal():
            reward += self.reward_dict["goal"]
            self.reward_dict["goal_scored"] = True
            info_dict["goal"] = True

        if self.opp_goal():
            reward -= self.reward_dict["goal"]
            self.reward_dict["goal_scored"] = True
            info_dict["opp_goal"] = True

        # # Attempting to kick
        # if action[3] > 0.8:
        #     if self.can_kick(agent):
        #         # Successful kick
        #         reward += self.reward_dict["kick"]
        #     else:
        #         # Missed kick
        #         reward += self.reward_dict["missed_kick"]

        # Ball to goal - Team
        # if (
        #     self.get_distance(self.prev_ball, [4800, 0])
        #     - self.get_distance(self.ball, [4800, 0])
        #     > 0
        # ):
        reward += self.reward_dict["ball_to_goal"] * (
            self.get_distance(self.prev_ball, [4800, 0])
            - self.get_distance(self.ball, [4800, 0])
            )

        # # Punish for hitting other robots
        # if self.contacting_robot(agent):
        #     reward += self.reward_dict["contact"]

        # # Punish for going out of bounds
        # if self.out_of_bounds(agent):
        #     reward += self.reward_dict["out_of_bounds"]

        return reward
