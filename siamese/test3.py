import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.decomposition import PCA
import sys
import os
from sklearn.manifold import TSNE
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model7 import EnhancedPainDataset
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt


# Assuming you have this already
dataset = EnhancedPainDataset("../movements.csv")  # or 'val' if you prefer
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Collect all features and labels
all_features = []
all_labels = []

for features, label in loader:
    all_features.append(features)
    all_labels.append(label)

# Stack all samples into arrays
X = torch.cat(all_features, dim=0)  # shape: [N, T, F]
if X.dim() == 3:
    X = X.mean(dim=1)  # shape becomes [N, F] — average across timesteps
X = X.cpu().numpy()
y = torch.cat(all_labels, dim=0).cpu().numpy()    # shape: [num_samples]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(6, 5))
plt.title("PCA Projection")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.7)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.show()

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(6, 5))
plt.title("t-SNE Projection")
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="coolwarm", alpha=0.7)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

kmeans = KMeans(n_clusters=2, random_state=42)
preds = kmeans.fit_predict(X)

ari = adjusted_rand_score(y.squeeze(), preds)
print("ARI score:", ari)

y = y.flatten()  # or y = y.ravel()

from umap import UMAP
X_umap = UMAP(n_components=2).fit_transform(X)

plt.figure(figsize=(8, 6))
plt.title("UMAP Projection")
plt.scatter(X_umap[y == 0, 0], X_umap[y == 0, 1], c='red', label='Fake', alpha=0.6)
plt.scatter(X_umap[y == 1, 0], X_umap[y == 1, 1], c='blue', label='Real', alpha=0.6)
plt.legend()
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True)
plt.show()