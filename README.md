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
to do the learning, such as Pseudo Maximum Likelihood Method, Score Matching, and Adaptive Cluter Expansion method among others.

In this implementation, the pseudo maximum likelihood method is used for learning the model. L-BFGS mehod is used for optimization and 
in the optimization, the weight matrix of the Potts model is restrainted to be symmetric. L2-normed penalty on the weight parameters is used for regulization.

# Reference
The implementation here mainly follows the method presented in reference 1. I also list the reference 2 and 3, where introduced pseduo maximum likelihood method and average-product correction method, respectively.

1. Ekeberg, Magnus, et al. "Improved contact prediction in proteins: using pseudolikelihoods to infer Potts models." Physical Review E 87.1 (2013): 012707.
2. Besag, Julian. "Statistical analysis of non-lattice data." The statistician (1975): 179-195.
3. Dunn, Stanley D., Lindi M. Wahl, and Gregory B. Gloor. "Mutual information without the influence of phylogeny or entropy dramatically improves residue contact prediction." Bioinformatics 24.3 (2007): 333-340.

# Example
