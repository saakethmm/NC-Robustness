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

plt.violinplot(info['distchange'],[1],showmeans=True,showmedians=False)

plt.violinplot(info3['distchange'],[2],showmeans=True,showmedians=False)

plt.violinplot(info2['distchange'],[3],showmeans=True,showmedians=False)

plt.violinplot(info2['dist2center'],[4],showmeans=True,showmedians=False)


# plt.violinplot(info['dist2center'],info['xlist'],showmeans=True,showmedians=False)
# plt.plot(info['xlist'],np.mean(np.array(info['dist2center']),axis=1), linewidth=3,label='dist2center')

# plt.violinplot(info['distchange'],info['xlist'],showmeans=True,showmedians=False)
# plt.plot(info['xlist'],np.mean(np.array(info['distchange']),axis=1), linewidth=3,label='original-dist2center-change')

# plt.violinplot(info3['distchange'],info['xlist'],showmeans=True,showmedians=False)
# plt.plot(info['xlist'],np.mean(np.array(info3['distchange']),axis=1), linewidth=4,label='esc-dist2center-change')

# plt.violinplot(info2['distchange'],info['xlist'],showmeans=True,showmedians=False)               
# plt.plot(info['xlist'],np.mean(np.array(info2['distchange']),axis=1), linewidth=3,label='multi-dist2center-change')


plt.legend()
plt.xlabel('attack_type', fontsize=15)
plt.ylabel('euclear_dist', fontsize=15)
plt.title('dist2center')


fig.savefig("dist-center.pdf", bbox_inches='tight')