Technical Summary
--------------------

**REEFFIT** core framework is essentially non-negative factorization with a Gaussian error model (i.e. a form of factor analysis). In this framework, the data are modeled as a linear combination of positive hidden variables:

.. math:: D_{obs} = WD + \epsilon 

Where :math:`D_{obs}` is the data, :math:`W` are the weights and :math:`D` are the hidden variables;  :math:`\epsilon` is a noise term that has mean 0 and position-wise variance values :math:`\Psi_i, \forall i=1,...n`.

In standard factor analysis, the hidden variables are normally distributed, and the likelihood function to obtain the weights and the noise covariance matrix is maximized using the EM algorithm. Fortunately, when the covariance matrix is assumed diagonal (that is, the measured variables are not well correlated), then the E and M steps of the EM algorithm can be written in closed form. However, we have a different, more complicated prior on the hidden variables :math:`D`, since in our case these variables are the chemical reactivities for a given structure that are expected to be drawn from a chemical reactivity distribution shaped by the structure. 

To simplify things, we use chemical reactivity distributions obtained from the `RMDB database <http://rmdb.stanford.edu>`_ splitted into two classes: distributions for unpaired and paired nucleotides. Because these priors on :math:`D` are far from normal, we cannot use the standard factor analysis EM-algorithm solutions; in fact, the likelihood function derived from the E-step cannot be calculated analytically. Instead, we either use Bayesian inference (Markov Chain Monte Carlo simulations) or maximum a posteriori estimation to solve the optimization problem. The M-step is much simpler, as it can be solved as a quadratic optimization problem with convex constraints (the weights need to be positive and sum to one). 

For error estimation, we use bootstrapping, although **REEFFIT** is able to calculate stnadard errors withouth bootstrapping as well.

