import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# l = np.load('../data/latent-8.npy')
# #p = np.load('../data/processed.npy')
# h = np.load('../data/haines.npy')

data = pd.read_csv('~/Desktop/columbia/capstone/fire-regimes/data/posneg.csv')
data = data[~np.isnan(data).any(axis=1)]
fire = data['fire']
data = data.drop(columns=['fire'])

random_indices = np.random.choice(data.index, size=10000, replace=False)
random_sample = data.loc[random_indices]
fire = fire.loc[random_indices]
# haines = h[random_indices]
#kmeans = KMeans(n_clusters=6,random_state=0).fit(random_sample)

tsne = TSNE(n_components=2, random_state=5)
X_tsne = tsne.fit_transform(random_sample)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=fire,alpha=0.4)
plt.colorbar()
plt.savefig('posneg-tsne.png')