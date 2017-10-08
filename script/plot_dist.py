__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/05/08 18:47:31"

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')

dist = []
with open("./output/Ras/dist_pdb.txt", 'r') as f:
    for l in f.readlines():
        l = l.strip()
        l = l.split(",")
        dist.append([int(l[0]), int(l[1]), float(l[2])])

idx = [ np.max([d[0], d[1]]) for d in dist]
num_res = np.max(idx)

dist_array = np.zeros((num_res, num_res))
for d in dist:
    i = d[0] - 1
    j = d[1] - 1
    dist_array[i,j] = d[2]
    dist_array[j,i] = d[2]

contact = dist_array <= 5
contact = 1 - contact

f = open("./output/Ras/seq_pos_idx.pkl", 'rb')
position_idx = np.array(pickle.load(f))
f.close()

contact = contact[position_idx,:][:,position_idx]

plt.imshow(contact, cmap = "gray")
plt.colorbar()
plt.savefig("./output/Ras/contact_pdb.pdf")
