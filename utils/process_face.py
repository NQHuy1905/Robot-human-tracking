import numpy as np

def normalize(emb):
    emb = np.squeeze(emb)
    norm = np.linalg.norm(emb)
    return emb / norm
