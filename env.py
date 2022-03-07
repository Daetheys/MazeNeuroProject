import numpy as np
import matplotlib.pyplot as plt
from pytz import NonExistentTimeError
import torch
from torch import nn
import collections
import gym

np.random.seed(42)

class Cell:
    def __init__(self,x=None,y=None):
        self.neighbours = [None,None,None,None] #left,down,up,right
        self.begin = False
        self.end = False
        self.x = x
        self.y = y
    def connect(self,cells):
        self.neighbours = cells
    def get_obs(self):
        return [float(not(n)) for n in self.neighbours]

class Maze:
    def __init__(self,size):
        self.size = size
    def generate(self):
        l_dir = [(-1,0),(0,-1),(0,1),(1,0)]
        def get_neighbours(i,j,fill=False):
            """ Returns a list of neighbours of the position"""
            l2 = []
            for (i2,j2) in l_dir:
                if i+i2>= 0 and i+i2<self.size[0] and j+j2>=0 and j+j2<self.size[1]:
                    l2.append((i+i2,j+j2))
                else:
                    if fill:
                        l2.append(None)
            return l2
        #Creates the grid
        self.grid = [[Cell(j,i) for i in range(self.size[1])] for j in range(self.size[0])]
        #Generates the maze
        init_pos = (np.random.randint(0,self.size[0]),np.random.randint(0,self.size[0]))
        blue = {(init_pos[0],init_pos[1]):None}
        black = {(init_pos[0],init_pos[1]):None}
        while len(blue) > 0:
            idx = np.random.randint(0,len(blue))
            blue_cell_coords = list(blue.keys())[idx]
            blue_cell = self.grid[blue_cell_coords[0]][blue_cell_coords[1]]
            legal_neighbours = get_neighbours(*blue_cell_coords,fill=True)
            flag = True
            for i,nei in enumerate(blue_cell.neighbours):
                if not(nei) and legal_neighbours[i]:
                    new_blue_cell_coords = (blue_cell_coords[0]+l_dir[i][0],blue_cell_coords[1]+l_dir[i][1])
                    try:
                        black[new_blue_cell_coords]
                        continue
                    except KeyError:
                        pass
                    new_blue_cell = self.grid[new_blue_cell_coords[0]][new_blue_cell_coords[1]]
                    blue_cell.neighbours[i] = new_blue_cell
                    new_blue_cell.neighbours[3-i] = blue_cell
                    blue[new_blue_cell_coords] = None
                    black[new_blue_cell_coords] = None
                    flag = False
                    break
            if flag:
                del blue[blue_cell_coords]

        #Set beginning and end of the maze
        begin = None
        end = None
        while begin == end:
            begin = self.random_cell()
            end = self.random_cell()
        self.begin_cell = begin
        self.begin_cell.begin = True

        self.end_cell = end
        self.end_cell.end = True

    def set_begin(self,cell):
        self.begin_cell.begin = False
        self.begin_cell = cell
        self.begin_cell.begin = True

    def set_end(self,cell):
        self.end_cell.end = False
        self.end_cell = cell
        self.end_cell.end = True

    def random_cell(self):
        i = np.random.randint(0,self.size[0])
        j = np.random.randint(0,self.size[1])
        return self.grid[i][j]
                
#Enforce unicity of the maze
class MazeEnv(gym.Env):
    max_time = 50
    def __init__(self,size=(5,5)):
        np.random.seed(42)
        self.maze = Maze(size)
        self.maze.generate()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0,1,(4,))

    def reset(self):
        self.maze.set_begin(self.maze.random_cell())
        self.timestep = 0
        self.position = self.maze.begin_cell
        return self.position.get_obs()

    def step(self,a):
        self.timestep += 1
        new_pos = self.position.neighbours[a]
        rew = -0.01
        if new_pos:
            self.position = new_pos
            if self.position.end:
                rew = 1
        return self.position.get_obs(),rew,(rew==1) or (self.timestep>self.max_time),{}

    def render(self):
        fig = plt.figure()
        for i in range(self.maze.size[0]):
            for j in range(self.maze.size[1]):
                if not(self.maze.grid[i][j].neighbours[0]):
                    plt.plot([i,i],[j,j+1],color='black',lw=1)
                if not(self.maze.grid[i][j].neighbours[3]):
                    plt.plot([i+1,i+1],[j,j+1],color='black',lw=1)
                if not(self.maze.grid[i][j].neighbours[1]):
                    plt.plot([i,i+1],[j,j],color='black',lw=1)
                if not(self.maze.grid[i][j].neighbours[2]):
                    plt.plot([i,i+1],[j+1,j+1],color='black',lw=1)
                if self.maze.grid[i][j] == self.position:
                    plt.scatter([i+0.5],[j+0.5],color='red',marker='x')
                if self.maze.grid[i][j].end:
                    plt.scatter([i+0.5],[j+0.5],color='green',marker='o')

class FixedMazeEnv(MazeEnv):
    def __init__(self,*args,**kwargs):
        super().__init__((12,12))