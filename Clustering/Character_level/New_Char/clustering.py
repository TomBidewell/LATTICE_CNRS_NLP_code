import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pickle

lang_embeds = torch.load("/home/tbidewell/home/POS_tagging/code/scripts/Clustering/Character_level/Pickled_Files/lang_embeds.pt")

with open("/home/tbidewell/home/POS_tagging/code/scripts/Clustering/Character_level/Pickled_Files/lang2id", "rb") as lang2id_fp:   #Pickling
    lang2id = pickle.load(lang2id_fp)

id2lang = []

for key, id in lang2id.items():
    id2lang.insert(id, key)


lang_embeds = lang_embeds.cpu().detach().numpy()

#remove UNK token
lang_embeds = np.delete(lang_embeds, lang2id['UNK'], axis = 0)

id2lang.remove('UNK')


# Perform hierarchical clustering using the linkage function=
X = linkage(lang_embeds, metric='cosine')  

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(X, labels= id2lang)
plt.xlabel('Languages')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()