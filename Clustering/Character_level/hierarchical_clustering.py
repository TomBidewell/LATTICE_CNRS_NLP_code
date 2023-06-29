import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pickle

lang_embeds = torch.load("lang_embeds.pt")

with open("lang2id", "rb") as lang2id_fp:   #Pickling
    lang2id = pickle.load(lang2id_fp)

with open("id2lang", "rb") as id2lang_fp:   #Pickling
    id2lang = pickle.load(id2lang_fp)



lang_embeds = lang_embeds.cpu().detach().numpy()

#remove UNK token
lang_embeds = np.delete(lang_embeds, lang2id['UNK'], axis = 0)

# Perform hierarchical clustering using the linkage function=
X = linkage(lang_embeds, metric='cosine')  # You can try different linkage methods

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(X, labels= id2lang)
plt.xlabel('Languages')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()