import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
import matplotlib.pyplot as plt
import pickle


with open("/home/tbidewell/home/POS_tagging/code/old/lang_counts_trial", "rb") as lang2id_fp:   #Pickling
    lang_embeds = pickle.load(lang2id_fp)

with open("/home/tbidewell/home/POS_tagging/code/old/id2lang_trial", "rb") as id2lang_fp:   #Pickling
    id2lang = pickle.load(id2lang_fp)



lang_embeds = lang_embeds / np.linalg.norm(lang_embeds, axis=1, keepdims=True)


# Perform hierarchical clustering using the linkage function=
X = linkage(lang_embeds, metric='cosine')  # You can try different linkage methods


with open("linkage_matrix_trial", "wb") as fp:
    pickle.dump(X, fp)





# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(X, labels= id2lang)
plt.xlabel('Languages')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
