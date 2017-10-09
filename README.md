# Description
This is a Tensorflow implementation of Potts models for Direct Coupling Analysis (DCA). 
Given a Multiple Sequence Alignment (MSA) for a protein, DCA is aimed to calculate the direct coupling between pairs of positions.
It can be used to predict the contact map of proteins using MSA and the predicted contact map is useful for 
predicting protein 3D structures.

One effective method for DCA is using Potts models, which are also called Boltzmann machines in machine learning.
Potts models belong to a larger category of models called generative probabilistic models, which means the model 
assigns a probablity for each sample and we can generate new samples by sampling from this distribution.
The first step when using these generative probabilistic models is to learn a model from observed data.
In the context of DCA, the observed data is a MSA and each sequence from the MSA is one sample.
As the Potts model assign probabilities for all the samples, theretically, we can write the likelihood
function of the data and use Maximum Likelihood Estimation (MLE) to learn the model.
In practice, the MLE method does not work well because it requires calcualting or estimating
the normalization constant of the model distribution. Several approximation methods have been developed
to do the learning, such as Pseudo Maximum Likelihood Method, Score Matching, and Adaptive Cluter Expansion among others.

