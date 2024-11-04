import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

l = np.load('latent.npy')

tsne = TSNE(n_components=2, random_state=1)
X_tsne = tsne.fit_transform(l[:5000])

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

plt.savefig('tsne_plot.png')