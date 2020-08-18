#KMeans Clustering


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import homogeneity_score
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np



#
#from sklearn import metrics
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix =metrics.cluster.contingency_matrix(y_true, y_pred)
# return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


df = pd.read_csv('2D_points2.txt', sep=" ", header=None, names=["x", "y"])

#plt.xlabel("X-component")
#plt.ylabel("Y-component")
#plt.scatter(df["x"],df["y"])


#---------------------------------------   Part-1   ------------------------------

print("----------------------------- Part-1 -----------------------\n")

km = KMeans(n_clusters=4, max_iter=100)
km.fit(df)
y_kmeans = km.predict(df)


labels = km.labels_

centers = km.cluster_centers_

plt.scatter(df['x'],df['y'], c=km.labels_, cmap='rainbow')


#
##  elbow method
#
#distortions = []
#K = range(1,10)
#for k in K:
#    kmeanModel = KMeans(n_clusters=k).fit(df)
#    kmeanModel.fit(df)
#    distortions.append(sum(min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])
#
## Plot the elbow
#plt.plot(K, distortions, 'bx-')
#plt.xlabel('k')
#plt.ylabel('Distortion')
#plt.title('The Elbow Method showing the optimal k')
#plt.show()

#Homogenity Score

homo = homogeneity_score(df['y'], y_kmeans)
print("Homogenity Score for K-MEANS: ",homo)

purity = purity_score(df['y'], y_kmeans)
print("Purity Score for KMeans : ",purity)
