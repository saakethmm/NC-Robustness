from cmath import inf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

f = open("info-pre.pkl",'rb')
info = pickle.load(f)

before_dict=info['before'][4]
pre_dict=info['pre'][4]

fig = plt.figure()
ax = fig.add_subplot(1,2,1,projection='3d') 
ax2 = fig.add_subplot(1,2,2,projection='3d') 
color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
acc=[]
x=[]
y=[]
z=[]
val=[]
ans=[]
# npbefore_dict=np.array(before_dict)
# print(type(npbefore_dict[0][:,0]))
# print(len(npbefore_dict[0][:,0]))
for cls in range (10):
    for idx in range(50):
        x.append(before_dict[cls][idx][0])
        y.append(before_dict[cls][idx][1])
        z.append( before_dict[cls][idx][2])
        ans.append(color[cls])
        nptemp=np.array(pre_dict[cls][idx])
        val.append(np.exp(nptemp[cls])/np.sum(np.exp(nptemp)))
        # ax.scatter(before_dict[cls][idx][0], before_dict[cls][idx][1], before_dict[cls][idx][2], \
        #     c=100*(math.exp(pre_dict[cls][idx][cls])/(math.exp(pre_dict[cls][idx][0])+math.exp(pre_dict[cls][idx][1])\
        #         +math.exp(pre_dict[cls][idx][2])+math.exp(pre_dict[cls][idx][3]))),cmap='plasma')
        acc.append(val)
        # if((math.exp(pre_dict[cls][idx][cls])/(math.exp(pre_dict[cls][idx][0])+math.exp(pre_dict[cls][idx][1])\
        #         +math.exp(pre_dict[cls][idx][2])+math.exp(pre_dict[cls][idx][3])))<0.5):
        #     ax.scatter(before_dict[cls][idx][0], before_dict[cls][idx][1], before_dict[cls][idx][2], \
        #     c=(math.exp(pre_dict[cls][idx][cls])/(math.exp(pre_dict[cls][idx][0])+math.exp(pre_dict[cls][idx][1])\
        #         +math.exp(pre_dict[cls][idx][2])+math.exp(pre_dict[cls][idx][3]))),cmap='plasma')
        # ax.scatter(before_dict[cls][idx][0], before_dict[cls][idx][1], before_dict[cls][idx][2], \
        #     c=color[cls],s=1)
ax.scatter(x,y,z, c=val,cmap='winter',s=1.4)
ax2.scatter(x,y,z, c=ans,s=1.4)
npacc=np.array(acc)
print(np.mean(npacc))
print(np.std(npacc))
print(npacc[npacc<0.5])
plt.show()
