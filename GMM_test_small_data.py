import matplotlib.pyplot as plt
import numpy as np
import sklearn
from mpl_toolkits import mplot3d
import Visualization

#todo: create data
nDimension = 3
nData = 4000
N = 1000
nCluster = 4
means = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0,-2.0,-3.0], [-2.0, 3.0, 4.0]])
covariances = np.array([np.diag([1.0,1.0,1.0]), np.diag([1.0,2.0,1.0]), np.diag([1.0,1.5,1.0]), np.diag([1.0,1.5,1.0])])
priors = np.array([0.25, 0.25, 0.25, 0.25])

X_train = []
for i in range(len(means)):
    x = np.random.multivariate_normal(means[i], covariances[i], N)
    X_train.append(x)

X_train = np.concatenate(X_train)
#print(X_train)

#todo: Visualize data
Visualization.visualize3D(X_train)

#todo: GMM
#init
def calculate_probability_density(X, means, covariances):
    probability_density = np.zeros((nData, nCluster))
    for i in range(nData):
        for k in range(nCluster):
            vector_2d = np.reshape((X[i] - means[k]), (nDimension, 1))
            a = np.exp(-0.5 * np.dot(np.dot(vector_2d.T, np.linalg.inv(covariances[k])), vector_2d)[0][0])
            b = np.sqrt(np.power(2 * np.pi, nDimension) * np.linalg.det(covariances[k]))
            probability_density[i][k] = a / b
    return probability_density

def calculate_probability_matrix(X, probability_density, priors):
    probability_matrix = np.zeros((nData, nCluster))
    for i in range(nData):
        px = 0
        for k in range(nCluster):
            px += priors[k] * probability_density[i][k]
        for k in range(nCluster):
            if (px == 0):
                probability_matrix[i][k] = 0;
            else:
                probability_matrix[i][k] = priors[k] * probability_density[i][k] / px

    return probability_matrix

def calculate_log_likelihood(probability_density, priors):
    log_likelihood = 0
    for i in range(nData):
        px = 0
        for k in range(nCluster):
            px += priors[k] * probability_density[i][k]
        if (px != 0): log_likelihood += np.log(px)
    return log_likelihood

def calculate_means(X, probability_matrix):
    means_new = np.zeros_like(means)
    for k in range(nCluster):
        Nk = 0
        for i in range(nData):
            Nk += probability_matrix[i][k]
        for i in range(nData):
            means_new[k] += probability_matrix[i][k] * X[i] / Nk
    return means_new

def calculate_covariances(X, probability_matrix, means):
    covariances_new = np.zeros_like(covariances)
    for k in range(nCluster):
        Nk = 0
        for i in range(nData):
            Nk += probability_matrix[i][k]
        for i in range(nData):
            vector_2d = np.reshape((X[i] - means[k]), (nDimension, 1))
            covariances_new[k] += probability_matrix[i][k] * np.dot(vector_2d, vector_2d.T) / Nk
    return covariances_new

def calculate_priors(X, probability_matrix):
    priors_new = np.zeros_like(priors)
    for k in range(nCluster):
        Nk = 0;
        for i in range(nData):
            Nk += probability_matrix[i][k]
        priors_new[k] = Nk / nData
    #print(probability_matrix[1123][0]); print("dm")
    return priors_new

def print_belonged_cluster(i):
    return np.argmax(probability_matrix[i])

convergenceCriterion = 0.1
preLogLikelihood = 0
curLogLikelihood = 0
count = 0
while (True):
    count += 1
    print(count)
    # E step
    probability_density = calculate_probability_density(X_train, means, covariances)
    probability_matrix = calculate_probability_matrix(X_train, probability_density, priors)
    # M step
    means = calculate_means(X_train, probability_matrix)
    covariances = calculate_covariances(X_train, probability_matrix, means)
    priors = calculate_priors(X_train, probability_matrix)
    # Evaluate step
    preLogLikelihood = curLogLikelihood
    curLogLikelihood = calculate_log_likelihood(probability_density, priors)
    print(preLogLikelihood, curLogLikelihood)
    if (curLogLikelihood - preLogLikelihood < convergenceCriterion and count > 10): break

print("Score found by my code: ", curLogLikelihood / nData)

#todo: GMM sklearn
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=nCluster, covariance_type='full')
gmm.fit(X_train)
print("Score found by sklearn: ", gmm.score(X_train))

