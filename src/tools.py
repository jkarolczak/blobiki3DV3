from abc import ABC

import numpy as np

from structs import FShape


class PromisingMotifsFinder:
    def __init__(self, motif: FShape, threshold: float = 0.8) -> None:
        self.motif = motif
        self.threshold = threshold

    def find(self, sequence: FShape, length: int) -> list[FShape]:
        promising_motifs = []
        for i in range(len(sequence) - length + 1):
            subsequence = sequence.fshape[i:i + length]
            if not np.isnan(subsequence).any() and (subsequence >= 1).any():
                promising_motifs.append(FShape(subsequence, sequence.acid[i:i + length]))
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


if __name__ == "__main__":
    from data import FShapeFileReader, FShapeDirectoryReader

    motif = FShapeFileReader("data/HNRNPC/hnrnpc_expected_pattern.txt").read()
    sequences = FShapeDirectoryReader("data/HNRNPC/hnrnpc_binding_sites_fshape").read()
    promising_motifs = PromisingMotifsFinder(motif).find_all(sequences[1])
    clusters = {length: DBScanClusterer().transform(motifs) for length, motifs in promising_motifs.items()}
    print(clusters)
