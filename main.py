import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import Visualization
import Optimize_nCluster
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.datasets import make_spd_matrix

# todo part1: load data
data = loadmat('cardio.mat')
X = np.array(data['X'])
y = data['y']
X_train = X[np.where(y == 0)[0]]
X_test = X[np.where(y == 1)[0]]
# Dimensionarity reduction: maintain 95% of original information. The result data shape is (1655, 13).
# Reason: To avoid overflow calculation
pca = PCA(0.95, whiten=True, random_state=0)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(X_test.shape)

# todo part2: Building my GMM
# part2.1: init parameters and variances: nData, nDimension, nClusters, means (mu), covariances (Sigma), priors (phi), logLikelihood
nData = X_train.shape[0]
nDimension = X_train.shape[1]
# Applying BIC to find optimal nCluster: 7
# nCluster = Optimize_nCluster.optimal_number_of_components(X_train)
nCluster = 7
# Applying K-means to initialize parameters
kmeans = KMeans(n_clusters=nCluster, random_state=0).fit(X_train)
means = kmeans.cluster_centers_  # init mu
priors = np.zeros(nCluster)
covariances = np.zeros((nCluster, nDimension, nDimension))      # using "full" covariance_type

for k in range(nCluster):
    Xk = X_train[np.where(kmeans.labels_ == k)[0]]
    priors[k] = float(Xk.shape[0]) / nData
    if np.size(Xk):
        covariances[k] = np.cov(Xk.T)       #Initialzie covariance matrices via points in each KMeans-cluster
    else:
        covariances[k] = np.cov(X_train.T)

# part2.2: Expectation-Maximization
def calculate_probability_density(X, means, covariances):
    probability_density = np.zeros((nData, nCluster))
    for i in range(X.shape[0]):
        for k in range(nCluster):
            vector_2d = np.reshape((X[i] - means[k]), (nDimension, 1))
            a = np.exp(-0.5 * np.dot(np.dot(vector_2d.T, np.linalg.inv(covariances[k])), vector_2d)[0][0])
            b = np.sqrt(np.power(2 * np.pi, nDimension) * np.linalg.det(covariances[k]))
            #if (i == 0 and k == 0) : print(np.linalg.det(covariances[0]))
            #print(i, np.power(2 * np.pi, nDimension) * np.linalg.det(covariances[k]))
            #print(i, np.linalg.det(covariances[k]))
            probability_density[i][k] = a / b
    return probability_density

def calculate_probability_matrix(X, probability_density, priors):
    probability_matrix = np.zeros((nData, nCluster))
    for i in range(X.shape[0]):
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
        Nk = 0;
        for i in range(nData):
            Nk += probability_matrix[i][k]
        for i in range(nData):
            vector_2d = np.reshape((X[i] - means[k]), (nDimension, 1))
            covariances_new[k] += probability_matrix[i][k] * np.dot(vector_2d, vector_2d.T)
        #print(Nk, np.linalg.det(covariances_new[k]))
        covariances_new[k] /= Nk
    #print(probability_density[0][0])
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
    # Convergence evaluation
    preLogLikelihood = curLogLikelihood
    curLogLikelihood = calculate_log_likelihood(probability_density, priors)

    print(curLogLikelihood / nData)
    if curLogLikelihood - preLogLikelihood < convergenceCriterion and count > 30: break

#todo part3: Compare with sklearn
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=nCluster, covariance_type='full', random_state=0)
gmm.fit(X_train)
print("Score found by my code: ", curLogLikelihood / nData)
print("Score found by sklearn: ", gmm.score(X_train))

#todo part3+ visualize samples with color in 3D
clt = [np.argmax(probability_matrix[i]) for i in range(nData)]      #the i sample belongs to cluster clt[i]
color_list = ['b', 'r', 'g', 'y', 'k', 'm', 'c']                    #color for 7 clusters
X_visualize = X[np.where(y == 0)[0]]
pca2 = PCA(3, whiten=True, random_state=0)
X_visualize = pca2.fit_transform(X_visualize)
x = X_visualize[:, 0]
y = X_visualize[:, 1]
z = X_visualize[:, 2]
ax = plt.axes(projection='3d')

#plotting all normal samples with different colors
#for i in range(nData):
#    ax.scatter3D(x[i], y[i], z[i], color=color_list[clt[i]])
#plt.show()

# todo part4: Anomaly detection
# calculate threshold = min probability among normal samples
prob = np.zeros((nData))
for i in range(nData):
    for k in range(nCluster):
      prob[i] += priors[k] * probability_density[i][k]
threshold = min(prob)

# calculate probability of suspect samples
probability_density_test = calculate_probability_density(X_test, means, covariances)
nTest = X_test.shape[0]
predict = np.zeros(nTest)
anomaly = np.zeros(nTest, dtype='bool')
anomaly_counter = 0
for i in range(nTest):
    for k in range(nCluster):
        predict[i] += priors[k] * probability_density_test[i][k]
    if (predict[i] < threshold):
        anomaly[i] = True
        anomaly_counter += 1

#plotting to visualize
for i in range(nData):
    ax.scatter3D(x[i], y[i], z[i], color='g')               #x,y,z from part3+
for i in range(nTest):
    if (anomaly[i]):
        ax.scatter3D(X_test[i][0], X_test[i][1], X_test[i][2], color='k')
print("Number of anomaly detected: ", anomaly_counter)
plt.show()