from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Using Bayesian information criterion (BIC) to find best model.
# Reason: stable condition (large and normal data) --> bic >> aic
def optimal_number_of_components(X):
    nCluster = np.arange(1, 31, 2)      #fix these parameters to find the convexhull
    gmms = [GaussianMixture(k, covariance_type='full', random_state=0) for k in nCluster]
    bics = [gmm.fit(X).bic(X) for gmm in gmms]
    plt.plot(nCluster, bics)
    plt.show()
    return nCluster[np.argmin(bics)]
