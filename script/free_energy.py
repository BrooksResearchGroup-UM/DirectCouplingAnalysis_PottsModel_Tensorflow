__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/07/31 20:47:54"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pickle
from sklearn import linear_model
import scipy.stats as stats
from sys import exit
import sys




## experimental data for folding free energy
exp_data = pd.read_csv('./free_energy/experiment/free_energy.csv')
with open("./output/aa_index.pkl", 'rb') as f:
    aa_index = pickle.load(f)

## information about the mutation: position idx, wild type aa, and mutation
exp_data['idx'] = None
exp_data['aa_wt'] = None
exp_data['aa_mu'] = None
for i in range(exp_data.shape[0]):
    d = exp_data.iloc[i, 1]
    exp_data.set_value(i, 'idx', d[1:-1])
    exp_data.set_value(i, 'aa_wt', d[0])
    exp_data.set_value(i, 'aa_mu', d[-1])


## idx of positions in potts model
with open('./output/seq_pos_idx.pkl', 'rb') as f:
    pos_idx = pickle.load(f)

wt_seq = "DAPSQIEVKDVTDTTALITWFKPLAEIDGIELTYGIKDVPGDRTTIDLTEDENQYSIGNLKPDTEYEVSLISRRGDMSSNP"
idx_offset = 3

## potts model parameters
correlation = []
pearsonr_max = 0
for k in range(26):
    weight_decay_value = k * 0.01
    with open("./output/model/pseudolikelihood_weight_decay_{:.2f}.pkl".format(weight_decay_value), 'rb') as f:
        model = pickle.load(f)

    J = model['J']
    h = model['h']
    exp_data['delta_G_potts'] = None
    for i in range(exp_data.shape[0]):
        idx = exp_data['idx'][i] ## position in pdb structure
        aa_wt = exp_data['aa_wt'][i]
        aa_mu = exp_data['aa_mu'][i]


        idx_seq = int(idx) - idx_offset ## position in wt_seq
        if idx_seq < 0:
            continue
        if idx_seq not in pos_idx:
            continue
        assert(aa_wt == wt_seq[idx_seq])

        idx_potts = pos_idx.index(idx_seq) ## position in potts model
        wt_ener = 0
        for k in range(len(pos_idx)):
            if k == idx_potts:
                continue
            wt_ener += -J[idx_potts*21 + aa_index[aa_wt], k*21 + aa_index[wt_seq[pos_idx[k]]]]
        wt_ener += -h[idx_potts*21 + aa_index[aa_wt]]

        mu_ener = 0
        for k in range(len(pos_idx)):
            if k == idx_potts:
                continue        
            mu_ener += -J[idx_potts*21 + aa_index[aa_mu], k*21 + aa_index[wt_seq[pos_idx[k]]]]
        mu_ener += -h[idx_potts*21 + aa_index[aa_mu]]

        exp_data.set_value(i, 'delta_G_potts', mu_ener - wt_ener)


    
    X = np.array(exp_data.delta_G_exp, dtype = np.float32)
    Y = np.array(exp_data.delta_G_potts, dtype = np.float32)
    flag = ~np.isnan(Y)
    X = X[flag]
    Y = Y[flag]
    
    # X = X.reshape((-1,1))
    # Y = Y.reshape((-1,1))
    # regression.fit(X, Y)

    pearsonr = stats.pearsonr(X,Y)[0]
    print("Weight decay: {0:.2f}, Pearson R: {1:.3f}".format(weight_decay_value, pearsonr))
    correlation.append(pearsonr)

    if pearsonr > pearsonr_max:
        pearsonr_max = pearsonr
        exp_data['delta_G_potts_best'] = exp_data.delta_G_potts

X = np.array(exp_data.delta_G_exp, dtype = np.float32)
Y = np.array(exp_data.delta_G_potts_best, dtype = np.float32)
flag = ~np.isnan(Y)
X = X[flag]
Y = Y[flag]

X = X.reshape((-1,1))
Y = Y.reshape((-1,1))
#Y = Y * 0.593
regression = linear_model.LinearRegression()
regression.fit(X, Y)

fig = plt.figure(0)
fig.clf()
ax = fig.add_subplot(111)
ax.plot(X, Y, 'bo')
tmpX = np.linspace(X.min()-1, X.max()+1, 20)
tmpX = tmpX.reshape((-1,1))
tmpY = regression.predict(tmpX)
ax.plot(tmpX, tmpY, 'k')
ax.text(0.2, 0.9, 'y = {0:.2f}*x + {1:.2f} \n R = {2:.2f}'.format(regression.coef_.item(), regression.intercept_.item(), pearsonr_max),
         verticalalignment='top', horizontalalignment='left',
         transform=ax.transAxes, fontsize = 14)
plt.xlim(X.min()-1.5, X.max()+1.5)
plt.ylim(Y.min()-1.5, Y.max()+1.5)
plt.xlabel('$\Delta\Delta G_{exp}$ (kcal/mol)', fontsize = 15)
plt.ylabel('$\Delta\Delta G_{potts}$', fontsize = 15)
plt.savefig('./output/free_energy.pdf')
#plt.show()
