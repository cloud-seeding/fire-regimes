import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

l = np.load('../data/latent-16.npy')[10000:25000]
# kmeans = KMeans(n_clusters=4,random_state=0).fit(l)

tsne = TSNE(n_components=2, random_state=1)
X_tsne = tsne.fit_transform(l)

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

plt.savefig('tsne_plot2d-noclust.png')