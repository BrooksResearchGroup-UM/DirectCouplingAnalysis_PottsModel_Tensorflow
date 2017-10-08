__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/05/15 01:42:31"

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser(description = "Calculate pairwise interaction scores infered from a Potts model and plot the top N interactions.")
parser.add_argument('model_file',
                    help = "path to a Potts model file")
parser.add_argument('N',
                    help = "the top N scored interactions are plotted",
                    type = int)
args = parser.parse_args()

######################################################################
#### Contact Maps from Potts Model (PLM) ######
######################################################################
## load model
with open(args.model_file, 'rb') as input_file_handle:
    model = pickle.load(input_file_handle)

len_seq = model['len_seq']
K = model['K']
num_node = model['num_node']
weight_decay_value = model['weight_decay']
maxiter = model['max_iter']
J = model['J']
h = model['h']

## calculate interaction scores
J_prime_dict = {}
score_FN = np.zeros([len_seq, len_seq])
for i in range(len_seq):
    for j in range(i+1, len_seq):
        J_prime = J[(i*K):(i*K+K), (j*K):(j*K+K)]
        J_prime = J_prime - J_prime.mean(0).reshape([1,-1]) - J_prime.mean(1).reshape([-1,1]) + J_prime.mean()
        J_prime_dict[(i,j)] = J_prime
        score_FN[i,j] = np.sqrt(np.sum(J_prime * J_prime))
        score_FN[j,i] = score_FN[i,j]
score_CN = score_FN - score_FN.mean(1).reshape([-1,1]).dot(score_FN.mean(0).reshape([1,-1])) / np.mean(score_FN)

for i in range(score_CN.shape[0]):
    for j in range(score_CN.shape[1]):
        if abs(i-j) <= 4:
            score_CN[i,j] = -np.inf
        
tmp = np.copy(score_CN).reshape([-1])
tmp.sort()
cutoff = tmp[-args.N*2]
contact_plm = score_CN > cutoff

######################################################################
#### Contact Maps from PDB Structure ######
######################################################################
protein = "Fibronectin_III"
cutoff = 8.5 ## cutoff distance for defining contact in PDB structure
offset = 2
dist = []
with open("./pdb/dist_pdb.txt", 'r') as f:
    for l in f.readlines():
        l = l.strip()
        l = l.split(",")
        dist.append([int(l[0]), int(l[1]), float(l[2])])
        
id_pdb = list(set([a[0] for a in dist] + [b[1] for b in dist] ))
id_pdb.sort()
num_res = len(id_pdb)
dist_array = np.zeros((num_res, num_res))
for d in dist:
    i = id_pdb.index(d[0])
    j = id_pdb.index(d[1])
    dist_array[i,j] = d[2]
    dist_array[j,i] = d[2]
    
contact_pdb = dist_array <= cutoff
for i in range(contact_pdb.shape[0]):
    for j in range(contact_pdb.shape[1]):
        if abs(id_pdb[i] - id_pdb[j]) <= 4:
            contact_pdb[i,j] = False

f = open("./pfam_msa/seq_pos_idx.pkl", 'rb')
position_idx = np.array(pickle.load(f))
f.close()

contact_pdb = contact_pdb[position_idx + offset,:][:,position_idx + offset]

##### Plot contacts from both #####
fig = plt.figure(figsize = (10,10))
fig.clf()
I,J = np.where(contact_pdb)
plt.plot(I,J, 'bo', alpha = 0.2, markersize = 8)
plt.axes().set_aspect('equal')
#plt.imshow(contact_pdb, cmap = "binary", alpha = 0.5)
I,J = np.where(contact_plm)
plt.plot(I,J, 'r^', markersize = 6, mew = 1.5)
# plt.xlim((0,153))
# plt.ylim((0,153))
plt.title(protein)
subprocess.run(['mkdir', '-p', 'output'])
plt.savefig("./output/contact_both.pdf")
plt.show()
sys.exit()
