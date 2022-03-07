import torch
import numpy as np

import matplotlib.pyplot as plt

from agent import agent
from env import FixedMazeEnv

env = FixedMazeEnv()
env.reset()

network_type = 'rnn'

env.render()

for j in range(len(env.maze.grid)):
        for k in range(len(env.maze.grid[j])):
            env.maze.grid[j][k].color = np.array([j*255/len(env.maze.grid),k*255/len(env.maze.grid),255])

ratio = 1/10
for i in range(100):
    for j in range(len(env.maze.grid)):
        for k in range(len(env.maze.grid[j])):
            for n in env.maze.grid[j][k].neighbours:
                if n:
                    n.color = n.color*(1-ratio)+env.maze.grid[j][k].color*ratio

for j in range(len(env.maze.grid)):
        for k in range(len(env.maze.grid[j])):
            c = np.int32(env.maze.grid[j][k].color)[None]/255
            plt.scatter([j+0.5],[k+0.5],marker='s',c=c)

agent.restore('checkpoint-1646606600\checkpoint_000150\checkpoint-150')

model = agent.get_policy().model

STATE = []
COLORS = []
ACTIONS = []
POSITIONS = []

R = 0
for k in range(100):
    obs = env.reset()
    seqlen = torch.tensor([1])
    state = model.get_initial_state()
    if network_type == 'lstm':
        state = [state[0][None],state[1][None]]
    else:
        state = [state[0][None]]
    #state = [state[0][None].cuda()]
    STATE.append([])
    COLORS.append([])
    ACTIONS.append([])
    POSITIONS.append([])
    for i in range(50):
        if network_type == 'lstm':
            STATE[k].append([state[0].detach().numpy(),state[1].detach().numpy()])
        else:
            STATE[k].append([state[0].detach().numpy()])
        COLORS[k].append(env.position.color)
        POSITIONS[k].append((env.position.x,env.position.y))
        #STATE[k].append([state[0].cpu().detach().numpy()])
        obs = {'obs':torch.tensor([obs])}
        pol,state = model(obs,state,seqlen)
        pol = torch.exp(pol)
        pol /= pol.sum()
        a = torch.argmax(pol)
        ACTIONS[k].append(a)
        obs,r,d,_ = env.step(a)
        R += r
        #env.render()
        if d:
            if network_type == 'lstm':
                STATE[k].append([state[0].detach().numpy(),state[1].detach().numpy()])
            else:
                STATE[k].append([state[0].detach().numpy()])
            COLORS[k].append(env.position.color)
            ACTIONS[k].append(-1)
            POSITIONS[k].append((env.position.x,env.position.y))
            #STATE[k].append([state[0].cpu().detach().numpy()])
            break
print(R)

states = [np.array(STATE[i])[...,0,:] for i in range(len(STATE))]

states_reshaped = np.concatenate(states,axis=0)

from sklearn.decomposition import PCA

pca = PCA(n_components=15)
if network_type == 'lstm':
    pca.fit(states_reshaped[:,0,:])
else:
    pca.fit(states_reshaped[:,0,0,:])

plt.figure()
plt.plot(range(pca.n_components_),pca.explained_variance_ratio_,marker='o')

if network_type == 'lstm':
    reshaped_projected_data = pca.transform(states_reshaped[:,0,:])
else:
    reshaped_projected_data = pca.transform(states_reshaped[:,0,0,:])
projected_data = []
p = 0
for i in range(len(states)):
    projected_data.append(reshaped_projected_data[p:p+len(states[i])])
    p += len(states[i])

from mpl_toolkits import mplot3d
from coloring import colorline
def plot_data(projected_data,mode): #action/position/time
    data = [projected_data[i][:] for i in range(len(projected_data))]
    reshaped_data = np.concatenate(data,axis=0)
    #fig = plt.figure(figsize=(20,20))
    ax = plt.axes(projection='3d')
    maxlen = max([len(projected_data[i]) for i in range(len(projected_data))])
    for i in range(len(projected_data)):
        alpha = 0.1
        lw = 1
        if i == 27:
            print(POSITIONS[i])
            alpha = 1
            lw = 2
        for j in range(len(projected_data[i])):
            if mode == 'time':
                c = np.array([[j/len(projected_data[i]),0,1-j/len(projected_data[i])]])
            elif mode == "position":
                c = np.int32(COLORS[i][j])/256
            elif mode == "action":
                colors = ['red','blue','green','purple','black']
                c = colors[ACTIONS[i][j]]
            elif mode == 'length':
                c = np.array([[len(projected_data[i])/maxlen,1-len(projected_data[i])/maxlen,0]])
            #ax.scatter3D([data[i][j][0]],[data[i][j][1]],[data[i][j][2]],c=c,marker='o',alpha=1)
            if j != len(data[i])-1:
                ax.plot3D([data[i][j][0],data[i][j+1][0]],[data[i][j][1],data[i][j+1][1]],[data[i][j][2]*0,data[i][j+1][2]*0],c=c,alpha=alpha,lw=lw)

    plt.xlim(reshaped_data[:,0].min(),reshaped_data[:,0].max())
    plt.ylim(reshaped_data[:,1].min(),reshaped_data[:,1].max())

plt.figure(figsize=(10,10))
plot_data(projected_data,'time')

plt.figure(figsize=(10,10))
plot_data(projected_data,'position')

plt.figure(figsize=(10,10))
plot_data(projected_data,'action')

plt.figure(figsize=(10,10))
plot_data(projected_data,'length')

plt.show()