# language_evolution

My code for a research paper with Mathieu Dehouck at CNRS started during summer internship. The paper will be finished soon.

Our aim was to demonstrate we could find the best evolutionary trees independent of model complexity, emphasising the value of older models.

I applied Multi-Task Learning to evolutionary language trees for Part of Speech (PoS) Tagging using bidirectional LSTMs; XLM-RoBERTa (transformer); and a 1-Dimensional CNN.

"Tree": stores the code for the models applied to various trees (Multi-Task Learning).

"All Languages": stores the models applied to all our languages individually (as a baseline score).

"Clustering": stores the code for generating two evolutionary trees using a Hierarchical Agglomerative Clustering aglorithm. One embeds languages using a Bag of Words representation for PoS trigrams from each language's dataset (reducing dimensionality by using SVD Decomposition). The second uses embeddings learnt through using an LSTM based next character generation model. Each character was concatenated with the character's respective language embedding before being fed into the LSTM. 

This internship significantly enhanced my skills in PyTorch, Python, parallelisation, multi-GPU usage, Linux, and independent research.

