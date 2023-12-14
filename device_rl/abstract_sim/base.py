import functools
import math
import time
from pettingzoo import ParallelEnv
import gymnasium as gym
import pygame
import numpy as np
import sys

sys.path.append(sys.path[0] + "/..")

'''
BaseEnv is a base class for all environments. It contains all the necessary
functions for a PettingZoo environment, but does not implement any of the
specifics of the environment. It is meant to be subclassed by other
environments.

Required:
- get_obs(self, agent)
- calculate_reward(self, agent)

Optional:
- reset(self, seed=None, return_info=False, options=None, **kwargs)
- __init__(self, continuous_actions=False, render_mode=None)


'''

class BaseEnv(ParallelEnv):
    metadata = {'render_modes': ['human', 'rgb_array']}

    '''
    Format for robots:
    x, y, angle

    Format for robot velocities:
    x, y, angle

    Format for ball:
    x, y, velocity, angle
    '''
    def __init__(self, render_mode='rgb_array'):
        '''
        Required:
        - possible_agents
        - action_spaces
        - observation_spaces
        '''
        self.rendering_init = False
        self.render_mode = render_mode

        
        # Other variables
        self.ball_radius = 10
        self.ball_acceleration = -0.2
        self.ball_velocity_coef = 1
        self.robot_radius = 25

        self.max_velocities = None

        self.episode_length = 2000
        self.last_touched = None


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def close(self):
        pass

    def reset(self, seed=None, return_info=False, options=None, **kwargs):
        raise NotImplementedError

    def get_obs(self, agent):
        """
        get_obs(agent) returns the observation for agent
        """
        raise NotImplementedError

    def calculate_reward(self, agent):
        """
        calculate_reward(agent) returns the reward for agent
        """
        raise NotImplementedError

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
        raise NotImplementedError

    #### STEP UTILS ####
    def clip_velocities(self, velocities):
        # Clip velocities
        # Positive x is 1, negative x is -0.3
        # Positive y is 0.5, negative y is -0.5
        # Angle clip is 0.5

        velocities[0] = np.clip(velocities[0], -0.3 * self.max_velocities['x'], 1 * self.max_velocities['x'])
        velocities[1] = np.clip(velocities[1], -0.5 * self.max_velocities['y'], 0.5 * self.max_velocities['y'])
        velocities[2] = np.clip(velocities[2], -0.5 * self.max_velocities['angle'], 0.5 * self.max_velocities['angle'])

        return velocities

    def move_agent(self, agent, action):
        i = self.agent_idx[agent]

        if action[3] > 0.8:
            # Kick the ball
            self.kick_ball(agent, opp=False)
        else:
            # Update velocities
            self.robot_velocities[i][0] += action[0] * self.x_disp
            self.robot_velocities[i][1] += action[1] * self.y_disp
            self.robot_velocities[i][2] += action[2] * self.angle_disp

            # Clip velocities
            self.robot_velocities[i] = self.clip_velocities(self.robot_velocities[i])

            policy_goal_x = self.robots[i][0] + (
            (
                (np.cos(self.robots[i][2]) * self.robot_velocities[i][0])
                + (np.cos(self.robots[i][2] + np.pi / 2) * self.robot_velocities[i][1])
            )
            * 100
            )  # the x component of the location targeted by the high level action
            policy_goal_y = self.robots[i][1] + (
                (
                    (np.sin(self.robots[i][2]) * self.robot_velocities[i][0])
                    + (np.sin(self.robots[i][2] + np.pi / 2) * self.robot_velocities[i][1])
                )
                * 100
            )  # the y component of the location targeted by the high level action

            # Update robot position
            self.robots[i][0] = (
                self.robots[i][0] * (1 - self.displacement_coef)
                + policy_goal_x * self.displacement_coef
            )  # weighted sums based on displacement coefficient
            self.robots[i][1] = (
                self.robots[i][1] * (1 - self.displacement_coef)
                + policy_goal_y * self.displacement_coef
            )  # the idea is we move towards the target position and angle

            # Update robot angle
            self.robots[i][2] = self.robots[i][2] + self.robot_velocities[i][2] * self.angle_displacement
            
            # Check for collisions with other robots from current action
            for other in self.agents:
                if other == agent:
                    continue
                self.check_collision(agent, other)
            for other in self.opponents:
                if other == agent:
                    continue
                self.check_collision(agent, other, opp2=True)
            
        # Make sure robot is on field
        self.robots[i][0] = np.clip(self.robots[i][0], -5200, 5200)
        self.robots[i][1] = np.clip(self.robots[i][1], -3700, 3700)

    def move_opponent(self, agent, action):
        i = self.opponent_idx[agent]

        if action[3] > 0.8:
            # Kick the ball
            self.kick_ball(agent, opp=True)
        else:
            # Update velocities
            self.opp_velocities[i][0] += action[0] * self.x_disp
            self.opp_velocities[i][1] += action[1] * self.y_disp
            self.opp_velocities[i][2] += action[2] * self.angle_disp

            # Clip velocities
            self.opp_velocities[i] = self.clip_velocities(self.opp_velocities[i])

            policy_goal_x = self.opp_robots[i][0] + (
            (
                (np.cos(self.opp_robots[i][2]) * self.opp_velocities[i][0])
                + (np.cos(self.opp_robots[i][2] + np.pi / 2) * self.opp_velocities[i][1])
            )
            * 100
            )  # the x component of the location targeted by the high level action
            policy_goal_y = self.opp_robots[i][1] + (
                (
                    (np.sin(self.opp_robots[i][2]) * self.opp_velocities[i][0])
                    + (np.sin(self.opp_robots[i][2] + np.pi / 2) * self.opp_velocities[i][1])
                )
                * 100
            )  # the y component of the location targeted by the high level action

            # Update robot position
            self.opp_robots[i][0] = (
                self.opp_robots[i][0] * (1 - self.displacement_coef)
                + policy_goal_x * self.displacement_coef
            )  # weighted sums based on displacement coefficient
            self.opp_robots[i][1] = (
                self.opp_robots[i][1] * (1 - self.displacement_coef)
                + policy_goal_y * self.displacement_coef
            )  # the idea is we move towards the target position and angle

            # Update robot angle
            self.opp_robots[i][2] = self.opp_robots[i][2] + self.opp_velocities[i][2] * self.angle_displacement
            
            # Check for collisions with other robots from current action
            for other in self.agents:
                if other == agent:
                    continue
                self.check_collision(agent, other, opp1=True)
            for other in self.opponents:
                if other == agent:
                    continue
                self.check_collision(agent, other, opp1=True, opp2=True)
            
        # Make sure robot is on field
        self.opp_robots[i][0] = np.clip(self.opp_robots[i][0], -5200, 5200)
        self.opp_robots[i][1] = np.clip(self.opp_robots[i][1], -3700, 3700)
        
        # Make sure ball is within bounds
        self.ball[0] = np.clip(self.ball[0], -5200, 5200)
        self.ball[1] = np.clip(self.ball[1], -3700, 3700)

    def dynamics_action_scale(self, action):
        # Action is a 4 dimensional vector, (angle, x, y, kick)
        # Unable to move if turning faster than 0.5
        if abs(action[0]) > 0.4:
            action[1] = 0
            action[2] = 0

        # Make moving backwards slower
        if action[1] < 0:
            action[1] *= 0.3

        # Make moving left and right slower
        action[2] *= 0.5

        return action
    
    # Check for collisions
    def check_collision(self, agent, other_agent, opp1=False, opp2=False):
        if not opp1:
            i = self.agent_idx[agent]
            robot_location = np.array([self.robots[i][0], self.robots[i][1]])
        else:
            i = self.opponent_idx[agent]
            robot_location = np.array([self.opp_robots[i][0], self.opp_robots[i][1]])

        if not opp2:
            j = self.agent_idx[other_agent]
            other_robot_location = np.array([self.robots[j][0], self.robots[j][1]])
        else:
            j = self.opponent_idx[other_agent]
            other_robot_location = np.array([self.opp_robots[j][0], self.opp_robots[j][1]])
        
        distance_robots = np.linalg.norm(other_robot_location - robot_location)

        # If collision, adjust velocities to bouce off each other
        bouce_multiplier = 10
        if distance_robots < (self.robot_radius + self.robot_radius) * 7:
            # Get angle between robots
            angle = math.atan2(other_robot_location[1] - robot_location[1], other_robot_location[0] - robot_location[0])
            # Get x and y components of velocity to bounce off each other
            x_vel = math.cos(angle) * bouce_multiplier
            y_vel = math.sin(angle) * bouce_multiplier

            # Update velocities and update positions
            if not opp1:
                self.robot_velocities[i][0] = -x_vel
                self.robot_velocities[i][1] = -y_vel

                self.robots[i][0] += self.robot_velocities[i][0]
                self.robots[i][1] += self.robot_velocities[i][1]
            else:
                self.opp_velocities[i][0] = -x_vel
                self.opp_velocities[i][1] = -y_vel

                self.opp_robots[i][0] += self.opp_velocities[i][0]
                self.opp_robots[i][1] += self.opp_velocities[i][1]
            if not opp2:
                self.robot_velocities[j][0] = x_vel
                self.robot_velocities[j][1] = y_vel

                self.robots[j][0] += self.robot_velocities[j][0]
                self.robots[j][1] += self.robot_velocities[j][1]
            else:
                self.opp_velocities[j][0] = x_vel
                self.opp_velocities[j][1] = y_vel

                self.opp_robots[j][0] += self.opp_velocities[j][0]
                self.opp_robots[j][1] += self.opp_velocities[j][1]

    def update_ball(self):
        # Update ball velocity
        self.ball[2] += self.ball_acceleration
        self.ball[2] = np.clip(self.ball[2], 0, 100)

        # Update ball position
        self.ball[0] += self.ball[2] * math.cos(self.ball[3])
        self.ball[1] += self.ball[2] * math.sin(self.ball[3])

        # If ball touches robot, push ball away
        for i in range(len(self.robots)):
            robot = self.robots[i]
            # Find distance between robot and ball
            robot_location = np.array([robot[0], robot[1]])
            ball_location = np.array([self.ball[0], self.ball[1]])
            distance_robot_ball = np.linalg.norm(ball_location - robot_location)

            # If collision, move ball away
            if distance_robot_ball < (self.robot_radius + self.ball_radius) * 6:
                self.ball[2] = self.ball_velocity_coef * 10
                self.ball[3] = math.atan2(self.ball[1] - robot[1], self.ball[0] - robot[0])

                self.ball[3] += np.random.normal(-1, 1) * np.pi/8
                self.last_touched = 'agent'

        # If ball touches opponent, push ball away
        for i in range(len(self.opp_robots)):
            robot = self.opp_robots[i]
            # Find distance between robot and ball
            robot_location = np.array([robot[0], robot[1]])
            ball_location = np.array([self.ball[0], self.ball[1]])
            distance_robot_ball = np.linalg.norm(ball_location - robot_location)

            # If collision, move ball away
            if distance_robot_ball < (self.robot_radius + self.ball_radius) * 6:
                self.ball[2] = self.ball_velocity_coef * 10
                self.ball[3] = math.atan2(self.ball[1] - robot[1], self.ball[0] - robot[0])

                self.ball[3] += np.random.normal(-1, 1) * np.pi/8
                self.last_touched = 'opp'
        

        # If ball it out of bounds, move ball onto OOB line and toward opponent goal
        # Get sign of ball as 1 or -1
        ball_x_sign = np.sign(self.ball[0])
        ball_y_sign = np.sign(self.ball[1])

        bouce = True
        wall_bounce_constant = 1

        if abs(self.ball[1]) > 3000:
            # Ball is out of bounds
            if bouce:
                self.ball[1] = ball_y_sign * 3000

                # Bounce ball off of "wall" by changing up and down velocity to be opposite
                self.ball[2] = self.ball[2] * wall_bounce_constant
                self.ball[3] = -self.ball[3]
            else:
                self.ball[1] = ball_y_sign * 3000
                if self.last_touched == 'agent':
                    # Move ball left
                    self.ball[0] -= 1000
                elif self.last_touched == 'opp':
                    # Move ball right
                    self.ball[0] += 1000
                else:
                    # Move ball toward center
                    self.ball[0] -= ball_x_sign * 1000

                self.ball[1] = 1500 * ball_y_sign
                self.ball[2] = 0
                self.ball[3] = 0

        # If ball is out of bounds at goal line, make corner kick. Ignoring ball in goal
        if abs(self.ball[0]) > 4500 and abs(self.ball[1]) > 1100:
            if bouce:
                self.ball[0] = ball_x_sign * 4500
                # Bounce ball off of "wall" by changing left and right velocity to be opposite
                self.ball[2] = self.ball[2] * wall_bounce_constant
                self.ball[3] = np.pi - self.ball[3]
            else:
                self.ball[0] = ball_x_sign * 3000
                self.ball[1] = ball_y_sign * 1500
                self.ball[2] = 0
                self.ball[3] = 0
            
    def check_facing_ball(self, agent, opp=False):
        if not opp:
            i = self.agent_idx[agent]
            # Convert from radians to degrees
            robot_angle = math.degrees(self.robots[i][2]) % 360

            # Find the angle between the robot and the ball
            angle_to_ball = math.degrees(
                math.atan2(self.ball[1] - self.robots[i][1], self.ball[0] - self.robots[i][0])
            )
        else:
            i = self.opponent_idx[agent]
            # Convert from radians to degrees
            robot_angle = math.degrees(self.opp_robots[i][2]) % 360

            # Find the angle between the robot and the ball
            angle_to_ball = math.degrees(
                math.atan2(self.ball[1] - self.opp_robots[i][1], self.ball[0] - self.opp_robots[i][0])
            )
        
        # Check if the robot is facing the ball
        req_angle = 10
        angle = (robot_angle - angle_to_ball) % 360

        if angle < req_angle or angle > 360 - req_angle:
            return True
        else:
            return False
        
    '''
    Gets relative position of object to agent
    '''
    def get_relative_observation(self, agent_loc, object_loc):
        # Get relative position of object to agent, returns x, y, angle
        # Agent loc is x, y, angle
        # Object loc is x, y

        # Get relative position of object to agent
        x = object_loc[0] - agent_loc[0]
        y = object_loc[1] - agent_loc[1]
        angle = np.arctan2(y, x) - agent_loc[2]

        # Rotate x, y by -agent angle
        xprime = x * np.cos(-agent_loc[2]) - y * np.sin(-agent_loc[2])
        yprime = x * np.sin(-agent_loc[2]) + y * np.cos(-agent_loc[2])

        return [xprime/10000, yprime/10000, np.sin(angle), np.cos(angle)]
    
    def kick_ball(self, agent, opp=False):
        if self.check_facing_ball(agent, opp=opp):
            if not opp:
                i = self.agent_idx[agent]
                robot_location = np.array([self.robots[i][0], self.robots[i][1]])
                self.last_touched = 'agent'
            else:
                i = self.opponent_idx[agent]
                robot_location = np.array([self.opp_robots[i][0], self.opp_robots[i][1]])
                self.last_touched = 'opp'

            # Find distance between robot and ball
            ball_location = np.array([self.ball[0], self.ball[1]])

            distance_robot_ball = np.linalg.norm(ball_location - robot_location)
            

            # If robot is close enough to ball, kick ball
            if distance_robot_ball < (self.robot_radius + self.ball_radius) * 10:
                self.ball[2] = 60
                # Set ball direction to be robot angle
                if not opp:
                    self.ball[3] = self.robots[i][2]
                else:
                    self.ball[3] = self.opp_robots[i][2]

                # self.ball_direction = math.atan2(self.ball[1] - robot_location[1], self.ball[0] - robot_location[0])

                
############ RENDERING UTILS ############

    def render_robot(self, agent):
        i = self.agent_idx[agent]
        render_length = 1200
        render_robot_x = int((self.robots[i][0] / 5200 + 1) * (render_length / 2))
        render_robot_y = int((self.robots[i][1] / 3700 + 1) * (render_length / 3))

        # Color = dark red 
        color = (140, 0, 0)

        # Draw robot
        pygame.draw.circle(
            self.field,
            pygame.Color(color[0], color[1], color[2]),
            (render_robot_x, render_robot_y),
            self.robot_radius,
            width=5,
        )

        # Draw robot direction
        pygame.draw.line(
            self.field,
            pygame.Color(50, 50, 50),
            (render_robot_x, render_robot_y),
            (
                render_robot_x + self.robot_radius * np.cos(self.robots[i][2]),
                render_robot_y + self.robot_radius * np.sin(self.robots[i][2]),
            ),
            width=5,
        )
        # Add robot number
        font = pygame.font.SysFont("Arial", 20)
        text = font.render(str(i), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (render_robot_x, render_robot_y)
        self.field.blit(text, textRect)

    def render_opponents(self, agent):
        # Pink = 255 51 255
        # Blue = 63 154 246

        i = self.opponent_idx[agent]
        render_length = 1200
        render_robot_x = int((self.opp_robots[i][0] / 5200 + 1) * (render_length / 2))
        render_robot_y = int((self.opp_robots[i][1] / 3700 + 1) * (render_length / 3))

        # Color = dark red 
        color = (63, 154, 246)

        # Draw robot
        pygame.draw.circle(
            self.field,
            pygame.Color(color[0], color[1], color[2]),
            (render_robot_x, render_robot_y),
            self.robot_radius,
            width=5,
        )

        # Draw robot direction
        pygame.draw.line(
            self.field,
            pygame.Color(50, 50, 50),
            (render_robot_x, render_robot_y),
            (
                render_robot_x + self.robot_radius * np.cos(self.opp_robots[i][2]),
                render_robot_y + self.robot_radius * np.sin(self.opp_robots[i][2]),
            ),
            width=5,
        )
        # Add robot number
        font = pygame.font.SysFont("Arial", 20)
        text = font.render(str(i), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (render_robot_x, render_robot_y)
        self.field.blit(text, textRect)

    '''
    Field dimensions are on page 2:
    https://spl.robocup.org/wp-content/uploads/SPL-Rules-2023.pdf
    '''
    def basic_field(self, _render_length=1200):
        render_length = _render_length
        
        # All dimensions are in mm proportional to the render_length
        # you can't change (based on the official robocup rule book ratio)
        

        # Total render length should be Border_strip_width * 2 + Field_length
        # Total render width should be Border_strip_width * 2 + Field_width

        # Field dimensions
        Field_length = 9000
        Field_width = 6000
        Line_width = 50
        Penalty_mark_size = 100
        Goal_area_length = 600
        Goal_area_width = 2200
        Penalty_area_length = 1650
        Penalty_area_width = 4000
        Penalty_mark_distance = 1300
        Center_circle_diameter = 1500
        Border_strip_width = 700

        # Create render dimensions
        Field_length_render = Field_length * render_length / (Field_length + 2 * Border_strip_width)
        Field_width_render = Field_width * render_length / (Field_length + 2 * Border_strip_width)
        Line_width_render = int(Line_width * render_length / (Field_length + 2 * Border_strip_width))
        Penalty_mark_size_render = Penalty_mark_size * render_length / (Field_length + 2 * Border_strip_width)
        Goal_area_length_render = Goal_area_length * render_length / (Field_length + 2 * Border_strip_width)
        Goal_area_width_render = Goal_area_width * render_length / (Field_length + 2 * Border_strip_width)
        Penalty_area_length_render = Penalty_area_length * render_length / (Field_length + 2 * Border_strip_width)
        Penalty_area_width_render = Penalty_area_width * render_length / (Field_length + 2 * Border_strip_width)
        Penalty_mark_distance_render = Penalty_mark_distance * render_length / (Field_length + 2 * Border_strip_width)
        Center_circle_diameter_render = Center_circle_diameter * render_length / (Field_length + 2 * Border_strip_width)
        Border_strip_width_render = int(Border_strip_width * render_length / (Field_length + 2 * Border_strip_width))
        Surface_width = int(Field_length_render + 2 * Border_strip_width_render)
        Surface_height = int(Field_width_render + 2 * Border_strip_width_render) - 40 # Constant here is just to make it look correct, unsure why it is needed

        Soccer_green = (18, 160, 0)
        self.field.fill(Soccer_green)

        # Draw out of bounds lines
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Border_strip_width_render, Border_strip_width_render),
            (Surface_width - Border_strip_width_render, Border_strip_width_render),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Border_strip_width_render, Surface_height - Border_strip_width_render),
            (Surface_width - Border_strip_width_render, Surface_height - Border_strip_width_render),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Border_strip_width_render, Border_strip_width_render),
            (Border_strip_width_render, Surface_height - Border_strip_width_render),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Surface_width - Border_strip_width_render, Border_strip_width_render),
            (Surface_width - Border_strip_width_render, Surface_height - Border_strip_width_render),
            width=Line_width_render,
        )

        # Draw center line
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Surface_width / 2, Border_strip_width_render),
            (Surface_width / 2, Surface_height - Border_strip_width_render),
            width=Line_width_render,
        )

        # Draw center circle
        pygame.draw.circle(
            self.field,
            pygame.Color(255, 255, 255),
            (int(Surface_width / 2), int(Surface_height / 2)),
            int(Center_circle_diameter_render / 2),
            width=Line_width_render,
        )

        # Draw center dot
        pygame.draw.circle(
            self.field,
            pygame.Color(255, 255, 255),
            (int(Surface_width / 2), int(Surface_height / 2)),
            int(Line_width_render / 2),
        )

        # Draw penalty areas
        # Left penalty area. Should be 1650mm long and 4000mm wide starting at goal line
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Border_strip_width_render, Surface_height / 2 - Penalty_area_width_render / 2),
            (Border_strip_width_render + Penalty_area_length_render, Surface_height / 2 - Penalty_area_width_render / 2),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Border_strip_width_render, Surface_height / 2 + Penalty_area_width_render / 2),
            (Border_strip_width_render + Penalty_area_length_render, Surface_height / 2 + Penalty_area_width_render / 2),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Border_strip_width_render + Penalty_area_length_render, Surface_height / 2 - Penalty_area_width_render / 2),
            (Border_strip_width_render + Penalty_area_length_render, Surface_height / 2 + Penalty_area_width_render / 2),
            width=Line_width_render,
        )

        # Right penalty area. Should be 1650mm long and 4000mm wide starting at goal line
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Surface_width - Border_strip_width_render, Surface_height / 2 - Penalty_area_width_render / 2),
            (Surface_width - Border_strip_width_render - Penalty_area_length_render, Surface_height / 2 - Penalty_area_width_render / 2),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Surface_width - Border_strip_width_render, Surface_height / 2 + Penalty_area_width_render / 2),
            (Surface_width - Border_strip_width_render - Penalty_area_length_render, Surface_height / 2 + Penalty_area_width_render / 2),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Surface_width - Border_strip_width_render - Penalty_area_length_render, Surface_height / 2 - Penalty_area_width_render / 2),
            (Surface_width - Border_strip_width_render - Penalty_area_length_render, Surface_height / 2 + Penalty_area_width_render / 2),
            width=Line_width_render,
        )

        # Draw goal areas
        # Left goal area. Should be 600mm long and 2200mm wide starting at goal line
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Border_strip_width_render, Surface_height / 2 - Goal_area_width_render / 2),
            (Border_strip_width_render + Goal_area_length_render, Surface_height / 2 - Goal_area_width_render / 2),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Border_strip_width_render, Surface_height / 2 + Goal_area_width_render / 2),
            (Border_strip_width_render + Goal_area_length_render, Surface_height / 2 + Goal_area_width_render / 2),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Border_strip_width_render + Goal_area_length_render, Surface_height / 2 - Goal_area_width_render / 2),
            (Border_strip_width_render + Goal_area_length_render, Surface_height / 2 + Goal_area_width_render / 2),
            width=Line_width_render,
        )

        # Right goal area. Should be 600mm long and 2200mm wide starting at goal line
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Surface_width - Border_strip_width_render, Surface_height / 2 - Goal_area_width_render / 2),
            (Surface_width - Border_strip_width_render - Goal_area_length_render, Surface_height / 2 - Goal_area_width_render / 2),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Surface_width - Border_strip_width_render, Surface_height / 2 + Goal_area_width_render / 2),
            (Surface_width - Border_strip_width_render - Goal_area_length_render, Surface_height / 2 + Goal_area_width_render / 2),
            width=Line_width_render,
        )
        pygame.draw.line(
            self.field,
            pygame.Color(255, 255, 255),
            (Surface_width - Border_strip_width_render - Goal_area_length_render, Surface_height / 2 - Goal_area_width_render / 2),
            (Surface_width - Border_strip_width_render - Goal_area_length_render, Surface_height / 2 + Goal_area_width_render / 2),
            width=Line_width_render,
        )

        # Draw penalty marks
        # Left penalty mark. Should be 100mm in diameter and 1300mm from goal line
        pygame.draw.circle(
            self.field,
            pygame.Color(255, 255, 255),
            (Border_strip_width_render + Penalty_mark_distance_render, Surface_height / 2),
            int(Penalty_mark_size_render / 2),
            width=Line_width_render,
        )

        # Right penalty mark. Should be 100mm in diameter and 1300mm from goal line
        pygame.draw.circle(
            self.field,
            pygame.Color(255, 255, 255),
            (Surface_width - Border_strip_width_render - Penalty_mark_distance_render, Surface_height / 2),
            int(Penalty_mark_size_render / 2),
            width=Line_width_render,
        )

        # Draw center point, same size as penalty mark
        pygame.draw.circle(
            self.field,
            pygame.Color(255, 255, 255),
            (Surface_width / 2, Surface_height / 2),
            int(Penalty_mark_size_render / 2),
            width=Line_width_render,
        )

        # Fill in goal areas with light grey, should mirror goal area flipped along goal line
        # Left goal area
        pygame.draw.rect(
            self.field,
            pygame.Color(255, 153, 153),
            (
                Border_strip_width_render - Goal_area_length_render,
                Surface_height / 2 - Goal_area_width_render / 2 - Line_width_render / 2,
                Goal_area_length_render,
                Goal_area_width_render,
            ),
        )

        # TODO: Make goal areas look better
        # Draw lines around goal areas
        # Left goal area
        # pygame.draw.line(
        #     self.field,
        #     pygame.Color(255, 255, 255),
        #     (Border_strip_width_render - Goal_area_length_render, Surface_height / 2 - Goal_area_width_render / 2),
        #     (Border_strip_width_render, Surface_height / 2 - Goal_area_width_render / 2),
        #     width=Line_width_render,
        # )
        # pygame.draw.line(
        #     self.field,
        #     pygame.Color(255, 255, 255),
        #     (Border_strip_width_render - Goal_area_length_render, Surface_height / 2 + Goal_area_width_render / 2),
        #     (Border_strip_width_render, Surface_height / 2 + Goal_area_width_render / 2),
        #     width=Line_width_render,
        # )


        # Right goal area
        pygame.draw.rect(
            self.field,
            pygame.Color(153, 204, 255),
            (
                Surface_width - Border_strip_width_render,
                Surface_height / 2 - Goal_area_width_render / 2 - Line_width_render / 2,
                Goal_area_length_render,
                Goal_area_width_render,
            ),
        )

    def render(self, mode="human"):
        render_length = 1200
        time.sleep(0.01)

        Field_length = 9000
        Field_width = 6000
        Border_strip_width = 700

        Field_length_render = Field_length * render_length / (Field_length + 2 * Border_strip_width)
        Field_width_render = Field_width * render_length / (Field_length + 2 * Border_strip_width)
        Border_strip_width_render = Border_strip_width * render_length / (Field_length + 2 * Border_strip_width)
        Surface_width = int(Field_length_render + 2 * Border_strip_width_render)
        Surface_height = int(Field_width_render + 2 * Border_strip_width_render)

        if self.rendering_init == False:
            pygame.init()

            self.field = pygame.display.set_mode((Surface_width, Surface_height))

            self.basic_field(render_length)
            pygame.display.set_caption("Point Targeting Environment")
            self.clock = pygame.time.Clock()

            self.rendering_init = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.basic_field(render_length)

        # Render robots
        for agent in self.agents:
            self.render_robot(agent)

        for agent in self.opponents:
            self.render_opponents(agent)

        # Render ball
        render_ball_x = int((self.ball[0] / 5200 + 1) * (render_length / 2))
        render_ball_y = int((self.ball[1] / 3700 + 1) * (render_length / 3))

        pygame.draw.circle(
            self.field,
            pygame.Color(40, 40, 40),
            (render_ball_x, render_ball_y),
            self.ball_radius,
        )

        pygame.display.update()
        self.clock.tick(60)

#########################################

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Return true if line segments AB and CD intersect, for goal line
    def intersect(A, B, C, D):
        def ccw(A,B,C):
            return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)