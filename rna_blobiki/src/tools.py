from abc import ABC
from collections import Counter

import numpy as np
import pandas as pd
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


class ConsensusMotifsFinder:

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


class SimilarMotifsFinder:
    def __init__(self, consensus_motifs: list[FShape], expected_motif: FShape):
        self.expected_motif = expected_motif
        self.consensus_motifs = consensus_motifs

    def compute_similar(self, motifs: list[FShape]) -> pd.DataFrame:
        similar_motifs = []
        for pattern in [self.expected_motif, *self.consensus_motifs]:
            for motif in motifs:
                matrix_profile = stumpy.stump(T_A=motif.fshape, m=len(pattern.fshape), T_B=pattern.fshape,
                                              ignore_trivial=False)
                indices = np.argwhere(matrix_profile[:, 0] <= 2.5).flatten()
                for ind in indices:
                    acid = motif.acid[ind:ind + len(pattern.fshape)]
                    fshape = motif.fshape[ind:ind + len(pattern.fshape)]
                    acids_same_len = self.compute_same_len(acid, self.expected_motif.acid)
                    fshapes_same_len = self.compute_same_len(fshape, self.expected_motif.fshape)
                    for a, f in zip(acids_same_len, fshapes_same_len):
                        zned = self.compute_zned(f, self.expected_motif.fshape)
                        ssf = self.compute_ssf(a, self.expected_motif.acid)
                        asd = 10 * zned - ssf
                        similar_motifs.append(
                            [''.join(acid), ''.join(a), (ind, ind + len(pattern.fshape) - 1), len(acid), motif.meta['file_name'], zned, ssf,
                             asd, ''.join(pattern.acid)])
        similar_motifs = pd.DataFrame(similar_motifs,
                                      columns=['new_motif', 'new_motif_same_len', 'nucleotides_range', 'len_new_motif', 'file', 'zned', 'ssf', 'as', 'consensus_motif'])
        similar_motifs = similar_motifs.sort_values(by=['as'])
        return similar_motifs

    @staticmethod
    def compute_same_len(similar_motif: np.ndarray, expected_motif: np.ndarray):
        if len(similar_motif) == len(expected_motif):
            return [similar_motif]
        elif len(similar_motif) - len(expected_motif) == 1:
            return [similar_motif[1:], similar_motif[:-1]]
        elif len(similar_motif) - len(expected_motif) == 2:
            return [similar_motif[1:-1]]

    @staticmethod
    def compute_zned(similar_motif_fshape: np.ndarray, expected_motif_fshape: np.ndarray) -> float:
        matrix_profile = stumpy.stump(T_A=similar_motif_fshape, m=len(similar_motif_fshape), T_B=expected_motif_fshape,
                                      ignore_trivial=False)
        return matrix_profile[0, 0]

    def compute_ssf(self, similar_motif_acid: np.ndarray, expected_motif_acid: np.ndarray) -> float:
        counter = 0
        for s, e in zip(similar_motif_acid, expected_motif_acid):
            if s == e or e == 'N':
                counter += 2
            elif s in ['A', 'G'] and e in ['A', 'G']:
                counter += 1
            elif s in ['C', 'U'] and e in ['C', 'U']:
                counter += 1
            elif s in ['T', 'C'] and e in ['T', 'C']:
                counter += 1
        return counter / len(expected_motif_acid)


if __name__ == "__main__":
    from data import FShapeFileReader, FShapeDirectoryReader

    motif = FShapeFileReader("data/HNRNPC/hnrnpc_expected_pattern.txt").read()
    sequences = FShapeDirectoryReader("data/HNRNPC/hnrnpc_binding_sites_fshape").read()
    promising_motifs = PromisingMotifsFinder(motif).find_all(sequences[1])
    clusters = {length: DBScanClusterer().transform(motifs) for length, motifs in promising_motifs.items()}
    print(clusters)
