import pickle
import matplotlib.pyplot as plt

f = open("info-roubustl2.pkl",'rb')
info = pickle.load(f)
f2 = open("info-roubust-nearest-escl2.pkl",'rb')
info2 = pickle.load(f2)
f3 = open("info-roubust-nearest-mutl2.pkl",'rb')
info3 = pickle.load(f3)

fig = plt.figure(figsize=(10, 8))

accs=[info['acc'][0],info2['acc'][0],info3['acc'][0]]
labels=['original-acc1','new-esc-acc1','new-mult-acc1']
bar_colors = ['tab:red', 'tab:blue', 'tab:orange']
plt.grid(True)
plt.bar(labels,accs, color=bar_colors)

plt.legend()
plt.xlabel('attack_type', fontsize=15)
plt.ylabel('accuracy', fontsize=15)
plt.title('robust')


fig.savefig("robust.pdf", bbox_inches='tight')