import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('~/Desktop/columbia/capstone/fire-regimes/data/profiles-areas.csv').drop(columns=['_uid_','initialdat','finaldate'])

# data = pd.concat([positives,negatives])
data = data[~np.isnan(data).any(axis=1)]
area = data['area_ha']
data = data.drop(columns=['area_ha'])


random_indices = np.random.choice(data.index, size=10000, replace=False)
random_sample = data.loc[random_indices]
area = np.log(area.loc[random_indices])
# haines = h[random_indices]
#kmeans = KMeans(n_clusters=6,random_state=0).fit(random_sample)

data_tensor = torch.tensor(random_sample.values, dtype=torch.float32)

# Run the data through the model to get the latent space
with torch.no_grad():
    latent_space = model(data_tensor).numpy()

tsne = TSNE(n_components=2, random_state=1)
X_tsne = tsne.fit_transform(latent_space)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=area,alpha=0.5)
plt.colorbar()
plt.savefig('plots/tsne-areas.png')