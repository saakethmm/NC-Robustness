import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cmath import inf
import pickle

TRACK_NUM=50
SCATTER_NUM=100

f = open("info-roubustl2.pkl",'rb')
info = pickle.load(f)
f2 = open("info-roubust-nearest-mutl2.pkl",'rb')
info2 = pickle.load(f2)
f3 = open("info-roubust-nearest-escl2.pkl",'rb')
info3 = pickle.load(f3)

before_dict=info2['mucdict'][0]
# before_dict=info2['mucdict'][0]
# pre_dict=info['pre'][4]

# def random_walk(num_steps, max_step=0.05):
#     """Return a 3D random walk as (num_steps, 3) array."""
#     start_pos = np.random.random(3)
#     steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
#     walk = start_pos + np.cumsum(steps, axis=0)
#     return walk


def update_lines(num, walks, lines):
    for line, walk in zip(lines, walks):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(walk[:num, :2].T)
        line.set_3d_properties(walk[:num, 2])
    return lines


# Data: 40 random walks as (num_steps, 3) arrays
num_steps = 10
walks=[]
walks2=[]
walks3=[]

# walks = [random_walk(num_steps) for index in range(40)]
npwalks= np.array(info['attackstep'][0])
for i in range(TRACK_NUM):
    walks.append(npwalks[:,i,:])
npwalks2= np.array(info2['attackstep'][0])
for i in range(TRACK_NUM):
    walks2.append(npwalks2[:,i,:])
npwalks3= np.array(info3['attackstep'][0])
for i in range(TRACK_NUM):
    walks3.append(npwalks3[:,i,:])

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(1,3,1,projection='3d') 
ax2 = fig.add_subplot(1,3,2,projection='3d') 
ax3 = fig.add_subplot(1,3,3,projection='3d') 

# Create lines initially without data
lines = [ax.plot([], [], [])[0] for _ in walks]
lines2 = [ax2.plot([], [], [])[0] for _ in walks2]
lines3 = [ax3.plot([], [], [])[0] for _ in walks3]

color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
x=[]
y=[]
z=[]
ans=[]

# Setting the axes properties
# ax.set(xlim3d=(0, 1), xlabel='X')
ax.set( xlabel='X')
ax.set( ylabel='Y')
ax.set( zlabel='Z')
ax.set( title='origin')
ax2.set( title='targeted')
ax3.set( title='escape')
ax2.set( xlabel='X')
ax2.set( ylabel='Y')
ax2.set( zlabel='Z')
ax3.set( xlabel='X')
ax3.set( ylabel='Y')
ax3.set( zlabel='Z')

for cls in range (10):
    for idx in range(SCATTER_NUM):
        x.append(before_dict[cls][idx][0])
        y.append(before_dict[cls][idx][1])
        z.append( before_dict[cls][idx][2])
        ans.append(color[cls])
# Creating the Animation object
ax.scatter(x,y,z, c=ans,s=1.4)
ax2.scatter(x,y,z, c=ans,s=1.4)
ax3.scatter(x,y,z, c=ans,s=1.4)
ani = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(walks, lines), interval=600)
ani2 = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(walks2, lines2), interval=600)
ani3 = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(walks3, lines3), interval=600)
plt.show()