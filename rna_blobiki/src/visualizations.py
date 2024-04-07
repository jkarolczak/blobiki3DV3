import numpy as np
from matplotlib import pyplot as plt

from structs import FShape


def visualize_statistics_clusters(clusters: np.ndarray) -> None:
    print(f'Number of clusters: {np.max(clusters)}')
    unique, counts = np.unique(clusters, return_counts=True)
    plt.bar(unique, counts, color='orange')
    plt.show()
    print('=========================================')


def visualize_motif(motif: FShape) -> None:
    print(f'Length of motif: {len(motif.fshape)}')
    print(f'FShape: {motif.fshape}')
    plt.plot(np.arange(0, len(motif.fshape)), motif.fshape, color='orange')
    plt.show()
    print('=========================================')
