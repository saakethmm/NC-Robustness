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
ax = fig.add_subplot()
plt.grid(True)

data=[info['realL2'][0],info3['realL2'][0],info2['realL2'][0]]
labelss=['original-realL2','esc-realL2','multi-realL2']
# plt.violinplot(data,showmeans=True,showmedians=False)
# set_axis_style(ax, labelss)
plt.violinplot(info['realL2'],[1],showmeans=True,showmedians=False)

plt.violinplot(info3['realL2'],[2],showmeans=True,showmedians=False)

plt.violinplot(info2['realL2'],[3],showmeans=True,showmedians=False)





# plt.legend()
plt.xlabel('attack_type', fontsize=15)
plt.ylabel('rate', fontsize=15)
plt.title('realL2')


fig.savefig("realL2.pdf", bbox_inches='tight')