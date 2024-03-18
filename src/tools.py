from abc import ABC
from collections import Counter

import numpy as np
import stumpy

from structs import FShape


class PromisingMotifsFinder:
    def __init__(self, motif: FShape, threshold: float = 1.0) -> None:
        self.motif = motif
        self.threshold = threshold

    def find(self, sequence: FShape, length: int) -> list[FShape]:
        promising_motifs = []
        for i in range(len(sequence) - length + 1):
            subsequence = sequence.fshape[i:i + length]
            subsequence_acid = sequence.acid[i:i + length]
            if not np.isnan(subsequence).any() and (subsequence >= self.threshold).any():
                promising_motifs.append(FShape(subsequence, subsequence_acid))
        return promising_motifs

    def find_all(self, sequence: FShape) -> dict[int, list[FShape]]:
        motif_length = len(motif)
        return {length: self.find(sequence, length) for length in range(motif_length, motif_length + 3)}


class IClusterer(ABC):
    def _prepare_motifs_for_clustering(self, motifs: list[FShape]) -> np.ndarray:
        return np.array([(motif.fshape - motif.fshape.mean()) / motif.fshape.std() for motif in motifs])

    def transform(self, motifs: list[FShape]) -> np.ndarray:
        fshapes = self._prepare_motifs_for_clustering(motifs)
        return self.clusterer.fit_predict(fshapes)


class DBScanClusterer(IClusterer):
    def __init__(self, eps: float = 2.5, min_samples: int = 1) -> None:
        from sklearn.cluster import DBSCAN

        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)


class AgglomerativeClusterer(IClusterer):
    def __init__(self, eps: int = 3) -> None:
        from sklearn.cluster import AgglomerativeClustering

        self.clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=eps)


class ConsensusMotifs:

    def _compute_largest_clusters(self, clusters: dict[int, list[int]]) -> list[tuple]:
        largest = []
        for key, cluster in clusters.items():
            cluster_freq = Counter(cluster).most_common(3)
            for c in cluster_freq:
                largest.append((key, c[0], c[1]))
        largest = sorted(largest, key=lambda x: x[2], reverse=True)[:3]
        largest = [i for i in largest if i[2] >= 3]
        return largest

    def compute_consensus(self, motifs: dict[int, list[FShape]], clusters: dict[int, list[int]]) -> list[FShape]:
        consensus_motifs = []
        largest_clusters = self._compute_largest_clusters(clusters)
        for cluster in largest_clusters:
            cluster_motifs = [m for i, m in enumerate(motifs[cluster[0]]) if clusters[cluster[0]][i] == cluster[1]]
            consensus = stumpy.ostinato([m.fshape for m in cluster_motifs], m=cluster[0])
            consensus = cluster_motifs[consensus[1]]
            consensus_motifs.append(consensus)
        return consensus_motifs


if __name__ == "__main__":
    from data import FShapeFileReader, FShapeDirectoryReader

    motif = FShapeFileReader("data/HNRNPC/hnrnpc_expected_pattern.txt").read()
    sequences = FShapeDirectoryReader("data/HNRNPC/hnrnpc_binding_sites_fshape").read()
    promising_motifs = PromisingMotifsFinder(motif).find_all(sequences[1])
    clusters = {length: DBScanClusterer().transform(motifs) for length, motifs in promising_motifs.items()}
    print(clusters)
