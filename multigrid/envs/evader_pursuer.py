from __future__ import annotations



from multigrid import MultiGridEnv
from multigrid.core import Grid
from multigrid.core.constants import Direction, Color
from multigrid.core.world_object import Goal

import random
import pygame

class GoalText(Goal):
    """
    Goal object an agent may be searching for.
    """

    def __new__(cls, color: str = Color.green):

        return super().__new__(cls, color=color)

    # def can_overlap(self) -> bool:
    #     """
    #     :meta private:
    #     """
    #     return True

    def render(self, img):
        """
        :meta private:
        """
        super().render(img)
        font_size = 10
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
        text = 0.0
        text_rect = font.get_rect(text, size=font_size)
        text_rect.center = img.get_rect().center
        text_rect.y = img.get_height() - font_size * 1.5
        font.render_to(img, text_rect, text, size=font_size)
        # fill_coords(img, point_in_rect(0, 1, 0, 1), self.color.rgb())

class PursuerEnv(MultiGridEnv):
    """
    .. image:: https://i.imgur.com/wY0tT7R.gif
        :width: 200

    ***********
    Description
    ***********

    This environment is an empty room, and the goal for each agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is subtracted
    for the number of steps to reach the goal.

    The standard setting is competitive, where agents race to the goal, and
    only the winner receives a reward.

    This environment is useful with small rooms, to validate that your RL algorithm
    works correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agents starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    *************
    Mission Space
    *************

    "get to the green goal square"

    *****************
    Observation Space
    *****************

    The multi-agent observation space is a Dict mapping from agent index to
    corresponding agent observation space.

    Each agent observation is a dictionary with the following entries:

    * image : ndarray[int] of shape (view_size, view_size, :attr:`.WorldObj.dim`)
        Encoding of the agent's partially observable view of the environment,
        where the object at each grid cell is encoded as a vector:
        (:class:`.Type`, :class:`.Color`, :class:`.State`)
    * direction : int
        Agent's direction (0: right, 1: down, 2: left, 3: up)
    * mission : Mission
        Task string corresponding to the current environment configuration

    ************
    Action Space
    ************

    The multi-agent action space is a Dict mapping from agent index to
    corresponding agent action space.

    Agent actions are discrete integer values, given by:

    +-----+--------------+-----------------------------+
    | Num | Name         | Action                      |
    +=====+==============+=============================+
    | 0   | left         | Turn left                   |
    +-----+--------------+-----------------------------+
    | 1   | right        | Turn right                  |
    +-----+--------------+-----------------------------+
    | 2   | forward      | Move forward                |
    +-----+--------------+-----------------------------+
    | 3   | pickup       | Pick up an object           |
    +-----+--------------+-----------------------------+
    | 4   | drop         | Drop an object              |
    +-----+--------------+-----------------------------+
    | 5   | toggle       | Toggle / activate an object |
    +-----+--------------+-----------------------------+
    | 6   | done         | Done completing task        |
    +-----+--------------+-----------------------------+

    *******
    Rewards
    *******

    A reward of ``1 - 0.9 * (step_count / max_steps)`` is given for success,
    and ``0`` for failure.

    ***********
    Termination
    ***********

    The episode ends if any one of the following conditions is met:

    * Any agent reaches the goal
    * Timeout (see ``max_steps``)

    """

    def __init__(
        self,
        size: int = 8,
        num_goals: int = 3,
        max_steps: int | None = None,
        joint_reward: bool = False,
        success_termination_mode: str = 'any',
        **kwargs):
        """
        Parameters
        ----------
        size : int, default=8
            Width and height of the grid
        agent_start_pos : tuple[int, int], default=(1, 1)
            Starting position of the agents (random if None)
        agent_start_dir : Direction, default=Direction.right
            Starting direction of the agents (random if None)
        max_steps : int, optional
            Maximum number of steps per episode
        joint_reward : bool, default=True
            Whether all agents receive the reward when the task is completed
        success_termination_mode : 'any' or 'all', default='any'
            Whether to terminate the environment when any agent reaches the goal
            or after all agents reach the goal
        **kwargs
            See :attr:`multigrid.base.MultiGridEnv.__init__`
        """

        self.agents_start_pos = [(size//2, size//2-4), (size//2, size//2)]
        self.agents_start_dir = [Direction.down, Direction.down]

        self.num_goals = num_goals
        self.goals = []
        self.goal = None

        super().__init__(
            mission_space="intercept the evader before it reaches the goal",
            grid_size=size,
            agents=2,
            max_steps=max_steps or (4 * size**2),
            joint_reward=joint_reward,
            success_termination_mode=success_termination_mode,
            **kwargs,
        )

        self.pursuer = self.agents[0]
        self.evader = self.agents[1]

    def reset(self):
        """
        Reset the environment
        """

        obs, info = super().reset()
        obs = self.mod_obs(obs)
        
        return obs, info

    def _gen_goals(self, num_goals):
        """
        Generate a list of goal positions in the grid
        """

        self.goals = []

        for i in range(num_goals):

            if i == 0:
                pos = self.place_obj(Goal(color=Color.blue))
                self.goal = pos
            else:
                pos = self.place_obj(Goal())
            
            self.goals.append(pos)

        
        # self.goal.COLOR = Color.blue

    def _gen_grid(self, width, height):
        """
        :meta private:
        """
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.grid.vert_wall(3, 2, 9)
        self.grid.vert_wall(6, 2, 9)
        self.grid.vert_wall(9, 2, 9)

        self.grid.vert_wall(12, 2, 9)

        # self.grid.vert_wall(3, 12, 8)
        self.grid.horz_wall(3, 12, 10)
        self.grid.horz_wall(3, 6, 4)
        self.grid.horz_wall(9, 6, 4)

        self._gen_goals(self.num_goals)

        # # Place a goal square in the bottom-right corner
        # self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        for i, agent in enumerate(self.agents):
            if self.agents_start_pos is not None and self.agents_start_dir is not None:
                agent.state.pos = self.agents_start_pos[i]
                agent.state.dir = self.agents_start_dir[i]
            else:
                self.place_agent(agent)

    def mod_obs(self, obs):
        # Pursuer
        mod_observations = {}
        mod_observations = [{"fov": obs[0]["image"], "grid": self.grid.state, 
                             "pos": self.pursuer.pos, "dir": self.pursuer.dir}]
        if 10 in obs[0]["image"]:
            mod_observations[0]["evader_pos"] = self.agents[1].pos
            mod_observations[0]["evader_dir"] = self.agents[1].dir

        # Evader
        mod_observations.append({"fov": obs[1]["image"], "grid": self.grid.state, 
                                 "pos": self.evader.pos, "dir": self.evader.dir})
        return mod_observations

    def is_done(self):
        """
        Check if the episode is done
        """

        base_done = super().is_done()
        
        # evader_done = self.agents[1]["pos"]==self.goal
        # pursuer_done = self.agents[0]["pos"]==self.agents[1]["pos"]

        # done = base_done or any(self.agent_states.terminated)

        print(self.evader.pos, self.pursuer.pos, self.goal)

        done = self.evader.pos == self.goal or self.agent_states.terminated[0]# self.pursuer.pos == self.goal

        return done
    
    def step(self, actions):
        """
        :meta private:
        """
        observations, rewards, terminations, truncations, infos = super().step(actions)
        observations = self.mod_obs(observations)
        
        return observations, rewards, terminations, truncations, infos