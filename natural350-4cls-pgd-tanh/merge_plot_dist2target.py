import pickle
import matplotlib.pyplot as plt
import numpy as np
import pdb

f = open("info-roubustl2.pkl",'rb')
info = pickle.load(f)
f2 = open("info-roubust-nearest-mutl2.pkl",'rb')
info2 = pickle.load(f2)
f3 = open("info-roubust-nearest-escl2.pkl",'rb')
info3 = pickle.load(f3)

fig = plt.figure(figsize=(10, 8))
plt.grid(True)

plt.violinplot(info['dist2targetchange'],[1],showmeans=True,showmedians=False)

plt.violinplot(info3['dist2targetchange'],[2],showmeans=True,showmedians=False)

plt.violinplot(info2['dist2targetchange'],[3],showmeans=True,showmedians=False)

plt.violinplot(info2['dist2target'],[4],showmeans=True,showmedians=False)
# plt.violinplot({'dist2target':info2['dist2target'],'original-dist2target-change':info['dist2targetchange'],'esc-dist2target-change':info3['dist2targetchange'],'multi-dist2target-change':info2['dist2targetchange']},showmeans=True,showmedians=False)
# plt.violinplot({'dist2target':info2['dist2target'],'original-dist2target-change':info['dist2targetchange'],'esc-dist2target-change':info3['dist2targetchange'],'multi-dist2target-change':info2['dist2targetchange']})

# plt.plot(info['xlist'],np.mean(np.array(info2['dist2target']),axis=1), linewidth=3,label='dist2target')

# plt.violinplot(info['dist2targetchange'],info['xlist'],showmeans=True,showmedians=False)
# plt.plot(info['xlist'],np.mean(np.array(info['dist2targetchange']),axis=1), linewidth=3,label='original-dist2target-change')

# plt.violinplot(info3['dist2targetchange'],info['xlist'],showmeans=True,showmedians=False)
# plt.plot(info['xlist'],np.mean(np.array(info3['dist2targetchange']),axis=1), linewidth=3,label='esc-dist2target-change')

# plt.violinplot(info2['dist2targetchange'],info['xlist'],showmeans=True,showmedians=False)               
# plt.plot(info['xlist'],np.mean(np.array(info2['dist2targetchange']),axis=1), linewidth=3,label='multi-dist2target-change')


plt.legend()
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('euclear_dist', fontsize=15)
plt.title('dist2target')


fig.savefig("dist-target.pdf", bbox_inches='tight')