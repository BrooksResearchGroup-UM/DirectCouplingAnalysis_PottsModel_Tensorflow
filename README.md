# Description
This is a Tensorflow implementation of Potts models for Direct Coupling Analysis (DCA). 
Given a Multiple Sequence Alignment (MSA) for a protein, DCA is aimed to calculate the direct coupling between pairs of positions.
It can be used to predict the contact map of proteins using MSA and the predicted contact map is useful for 
predicting protein 3D structures.

One effective method for DCA is using Potts models, which are also called Boltzmann machines in machine learning.
Potts models belong to a larger category of models called generative probablistic models, which means the model 
assigns a probablity for each sample and we can generate new samples by sampling from this distribution.
Before we can sample from the model, we need to learn a model based on the observed data.

In the context of DCA, the observed data is a MSA and each sequence from the MSA is one sample.
