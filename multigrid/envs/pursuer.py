from __future__ import annotations

from multigrid import MultiGridEnv
from multigrid.core import Grid
from multigrid.core.constants import Direction, Color, IDX_TO_COLOR
from multigrid.core.world_object import Goal, Wall

import random
import pygame
import numpy as np

DIRECTIONS = [[np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])],
              [np.array([-1, 0]), np.array([-1, 1]), np.array([0, 2]), np.array([1, 2]), 
                np.array([2, 1]), np.array([2, 0]), np.array([0, -1]), np.array([1, -1])
               ]
              ]

def get_neighbours(coord, size, cell_size):
    neighbours = [coord+dir for dir in DIRECTIONS[cell_size-1] 
                  if np.max(coord+dir) < size and np.min(coord+dir) >= 0]
    return neighbours

def unfill(grid, coord, size, cell_size):
    x1 = np.clip(coord[0], 1, size-1)
    x2 = np.clip(x1+cell_size, 1, size-1)
    y1 = np.clip(coord[1], 1, size-1)
    y2 = np.clip(y1+cell_size, 1, size-1)
    grid[x1:x2, y1:y2] = 0
    return grid

def generate_base_grid(size, target_pos, cell_size=1):

    grid = np.ones((size, size), dtype=int)
    grid[target_pos[0]:target_pos[0]+1, target_pos[1]-4:target_pos[1]+4] = 0
    grid[target_pos[0]-4:target_pos[0]+4, target_pos[1]:target_pos[1]+1] = 0


    #Choose 2 random points
    start = np.random.randint(0, size//cell_size, 2)
    grid = unfill(grid, start, size, cell_size)
    explored = {tuple(start): 1}
    queue = get_neighbours(start, size, cell_size)

    while len(queue)>0:
        
        # idx = random.randint(0, len(queue)-1)
        idx = 3 if len(queue)>3 else 0
        cell = queue[idx]
        explored[tuple(cell)] = 1
        neighbours = get_neighbours(cell, size, cell_size)
        filled_neighbours = [neighbour for neighbour in neighbours 
                            if grid[tuple(neighbour)] == 1]

        # The cell doesn't have 2 explored neighbours
        if ((cell_size==1) and (len(filled_neighbours) > 2) or (cell_size==2) and (len(filled_neighbours) > 2)):
            # grid[tuple(cell)] = 0
            grid = unfill(grid, cell, size, cell_size)
            queue += [neighbour for neighbour in filled_neighbours
                    if tuple(neighbour) not in explored]
            
        queue.pop(idx)
        # Change the cell size randomly
        cell_size = random.randint(1, 2) if cell[0]%2==0 and cell[1]%2==0 else 1

    return grid

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
        size: int | None = 8,
        base_grid: np.array | None = None,
        goals: list[tuple[int, int]] | None = None,
        target_pos: tuple[int, int] | None = None,
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

        if base_grid is not None:
            size = base_grid.shape[0]

        if target_pos is not None:
            observer_pos = target_pos[0], target_pos[1]-4
            self.agents_start_pos = [observer_pos, target_pos]
            self.agents_start_dir = [Direction.down, Direction.down]
        else:
            self.agents_start_pos = [(size//2, size//2-4), (size//2, size//2)]
            self.agents_start_dir = [Direction.down, Direction.down]

        self.base_grid = base_grid

        self.num_goals = num_goals

        self.goals = []
        self.goal = None
        if goals is not None:
            self.goals = goals
            self.goal = goals[0]
        

        super().__init__(
            mission_space="intercept the target before it reaches the goal",
            grid_size=size,
            agents=2,
            max_steps=max_steps or (4 * size**2),
            joint_reward=joint_reward,
            success_termination_mode=success_termination_mode,
            **kwargs,
        )

        self.observer = self.agents[0]
        self.target = self.agents[1]

        self.POS2COLOR = {}

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


        for i in range(num_goals):

            obj = Goal(IDX_TO_COLOR[i])
            if len(self.goals) == num_goals:
                pos = self.goals[i]
                self.grid.set(pos[0], pos[1], obj)
            else:
                pos = self.place_obj(obj)
                self.goals.append(pos)

            if i == 0:
                self.goal = pos

            self.POS2COLOR[pos] = str(IDX_TO_COLOR[i]).split(".")[1]            

        
        # self.goal.COLOR = Color.blue

    def _gen_grid(self, width, height):
        """
        :meta private:
        """
        # Create an empty grid
        if self.base_grid is None:
            self.base_grid = generate_base_grid(width, self.agents_start_pos[1])
            
        width, height = self.base_grid.shape
        self.grid = Grid(width, height)

        rows, cols = np.where(self.base_grid==1)
        self.grid.state[rows, cols] = Wall()

        self._gen_goals(self.num_goals)

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
                             "pos": self.observer.pos, "dir": self.observer.dir}]
        if 10 in obs[0]["image"]:
            mod_observations[0]["target_pos"] = self.agents[1].pos
            mod_observations[0]["target_dir"] = self.agents[1].dir

        # Target
        mod_observations.append({"fov": obs[1]["image"], "grid": self.grid.state, 
                                 "pos": self.target.pos, "dir": self.target.dir})
        return mod_observations

    def is_done(self):
        """
        Check if the episode is done
        """

        base_done = super().is_done()
        done = self.target.pos == self.goal or self.agent_states.terminated[0]# self.observer.pos == self.goal

        return done
    
    def step(self, actions):
        """
        :meta private:
        """
        observations, rewards, terminations, truncations, infos = super().step(actions)
        observations = self.mod_obs(observations)
        
        return observations, rewards, terminations, truncations, infos