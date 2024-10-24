import os
import sys
import torch
import numpy as np
import pyroved as pv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

data = pd.read_csv('~/Desktop/columbia/capstone/fire-regimes/data/merged_data.csv')

df_dedup = data.groupby(['time', 'x', 'y', 'level'], as_index=False).mean()
df_pivoted = df_dedup.set_index(['time', 'y', 'x', 'level']).unstack('level')

df_pivoted.columns = [f'{level}-{var}' for var, level in df_pivoted.columns]
df = df_pivoted.reset_index().drop(columns=['time','y','x'])

X = df[~np.isnan(df).any(axis=1)]
X = normalize(X,axis=0)

train_data = torch.from_numpy(X).float()
train_loader = pv.utils.init_dataloader(train_data, batch_size=64)
in_dim = (174,)

latent_dims = [2,4,8,16,32]

def train(n_epochs, latent_dim):

    save_path = f"vae-{latent_dim}.pth"
    vae = pv.models.iVAE(in_dim, latent_dim=latent_dim, invariances=None, seed=0)
    trainer = pv.trainers.SVItrainer(vae)

    for e in range(n_epochs):
        trainer.step(train_loader)
        trainer.print_statistics()

    torch.save(vae.state_dict(), save_path)

def main():

    if torch.cuda.is_available():
        print("GPU is available!")
    else:
        print("Something's wrong; cannot detect GPU")

    for latent_dim in latent_dims:
        train(100, latent_dim)