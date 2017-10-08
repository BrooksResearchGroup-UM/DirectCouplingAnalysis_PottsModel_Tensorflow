__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/05/13 23:43:38"

import numpy as np
import pickle
import tensorflow as tf
from sys import exit
import sys
import timeit
import argparse
import subprocess

parser = argparse.ArgumentParser(description = "Learn a Potts model using Multiple Sequence Alignment data.")
parser.add_argument("input_dir",
                    help = "input directory where the files seq_msa_binary.pkl, seq_msa.pkl, seq_weight.pkl are.")
parser.add_argument("max_iter",
                    help = "The maximum num of iteratioins in L-BFGS optimization.",
                    type = int)
parser.add_argument("weight_decay",
                    help = "weight decay factor of L2 penalty",
                    type = float)
parser.add_argument("output_dir",
                    help = "output directory for saving the model")
args = parser.parse_args()

## read msa
msa_file_name = args.input_dir + "/seq_msa_binary.pkl"
with open(msa_file_name, 'rb') as input_file_handle:
    seq_msa_binary = pickle.load(input_file_handle)

msa_file_name = args.input_dir + "/seq_msa.pkl"
with open(msa_file_name, 'rb') as input_file_handle:
    seq_msa = pickle.load(input_file_handle)

weight_file_name = args.input_dir + "/seq_weight.pkl"
with open(weight_file_name, 'rb') as input_file_handle:
    seq_weight = pickle.load(input_file_handle)


## pseudolikelihood method for Potts model
_, len_seq, K = seq_msa_binary.shape
num_node = len_seq * K
batch_size = tf.placeholder(tf.int32)
data = tf.placeholder(tf.float32, shape = [None, num_node])
data_weight = tf.placeholder(tf.float32, [None])
half_J = tf.Variable(tf.zeros([num_node, num_node]))
h = tf.Variable(tf.zeros([num_node]))
J = half_J + tf.transpose(half_J)
J_mask_value = np.ones((num_node, num_node), dtype = np.float32)
for i in range(len_seq):
    J_mask_value[K*i:K*i+K, K*i:K*i+K] = 0
J_mask = tf.constant(J_mask_value)
J = J * J_mask
logits = tf.matmul(data, J) + h
logits = tf.reshape(logits, [-1, K])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits( logits = logits, labels = tf.reshape(data, [-1,K]))
cross_entropy = tf.reduce_sum(tf.reshape(cross_entropy, [-1, len_seq]), axis = 1)
cross_entropy = tf.reduce_sum(cross_entropy * data_weight)
weight_decay = tf.placeholder(tf.float32)
cross_entropy = cross_entropy + weight_decay * tf.reduce_sum(tf.square(J))

## create a session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

## trainging using L-BFGS-B algorithm
feed_dict = {data: seq_msa_binary.reshape((-1,num_node)), data_weight: seq_weight, weight_decay: args.weight_decay}
print("Initial Cross Entropy: ", sess.run(cross_entropy, feed_dict = feed_dict))
start_time = timeit.time.time()
optimizer = tf.contrib.opt.ScipyOptimizerInterface(cross_entropy, var_list = [half_J,h], method = "L-BFGS-B", options={'maxiter': args.max_iter, 'disp': 1})
optimizer.minimize(sess, feed_dict = feed_dict)
end_time = timeit.time.time()
print("Time elapsed in secondes: ", end_time - start_time)

## save J and h
model = {}
model['len_seq'] = len_seq
model['K'] = K
model['num_node'] = num_node
model['weight_decay'] = args.weight_decay
model['max_iter'] = args.max_iter
model['J'] = sess.run(J)
model['h'] = sess.run(h)

subprocess.run(['mkdir', '-p', args.output_dir])
with open("{}/model_weight_decay_{:.3f}.pkl".format(args.output_dir, args.weight_decay), 'wb') as output_file_handle:
    pickle.dump(model, output_file_handle)
