from __future__ import annotations

import enum
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xmltodict


@dataclass
class Motif:
    df: pd.DataFrame
    rmsd: float
    name: str
    motif_name: str
    puzzle_name: str

    @staticmethod
    def read(rmsd: float, name: str, motif_name: str, puzzle_name: str) -> Motif:
        df = pd.read_csv((Path("data") / puzzle_name / motif_name / name).with_suffix(".tor"), sep="\t")
        return Motif(df, rmsd, name, motif_name, puzzle_name)

    @property
    def alpha(self) -> np.ndarray:
        return self.df["alpha"].values

    @property
    def beta(self) -> np.ndarray:
        return self.df["beta"].values

    @property
    def gamma(self) -> np.ndarray:
        return self.df["gamma"].values

    @property
    def delta(self) -> np.ndarray:
        return self.df["delta"].values

    @property
    def epsilon(self) -> np.ndarray:
        return self.df["epsilon"].values

    @property
    def zeta(self) -> np.ndarray:
        return self.df["zeta"].values

    @property
    def chi(self) -> np.ndarray:
        return self.df["chi"].values

    @property
    def angles(self) -> np.ndarray:
        values = self.df[["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]].values
        values = np.where(values == "-", 0.0, values)
        values = values.astype(np.float32)
        return values


@dataclass
class Backbone:
    motifs: list[Motif]
    name: str
    puzzle_name: str
    segments: int
    residues: int
    ranges: str
    sequences: str

    @staticmethod
    def read(name: str, puzzle_name: str, segments: int, residues: int, ranges: str, sequences: str) -> Backbone:
        motifs = []

        if (path := (Path("data") / puzzle_name / f"{name}-rmsd").with_suffix(".xml")).exists():
            with path.open("r") as fp:
                xml = xmltodict.parse(fp.read())

            rmsds = {item["description"]["filename"].replace(".pdb", ""): item["score"]
                     for item in xml["measureScores"]["structure"]}

            for tor in (Path("data") / puzzle_name / name).glob("*.tor"):
                tor_name = tor.stem
                motifs.append(Motif.read(float(rmsds[tor_name]), tor_name, name, puzzle_name))
            return Backbone(motifs, name, puzzle_name, segments, residues, ranges, sequences)


@dataclass
class Puzzle:
    backbones: list[Backbone]
    name: str

    @staticmethod
    def read(name: str) -> Puzzle:
        filter_results = pd.read_csv(
            Path("data") / name / "filter-results.txt",
            sep="\t",
            names=("motif", "segments", "residues", "ranges", "sequences")
        )
        filter_results = filter_results[filter_results["segments"] >= 3].reset_index(drop=True)
        motifs = []
        for _, row in filter_results.iterrows():
            if motif := Backbone.read(row["motif"], name, row["segments"], row["residues"], row["ranges"], row["sequences"]):
                motifs.append(motif)
        return Puzzle(motifs, name)


class Dataset:
    def __init__(self, motifs: list[Motif], device: str = "cpu", random_seed: int = 42) -> None:
        self.motifs = motifs
        self.device = device

        random.seed(random_seed)
        random.shuffle(self.motifs)

    @property
    def mean(self) -> float:
        return np.mean([motif.rmsd for motif in self.motifs])

    @property
    def std(self) -> float:
        return np.std([motif.rmsd for motif in self.motifs])

    def __len__(self) -> int:
        return len(self.motifs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        motif = self.motifs[idx]
        return torch.tensor(motif.angles, device=self.device), torch.tensor([motif.rmsd], device=self.device)


class Strategy(enum.IntEnum):
    RANDOM = enum.auto()
    BY_PUZZLE = enum.auto()


def _read_puzzles() -> list[Puzzle]:
    return [Puzzle.read(f"pz{i:0>2}") for i in range(1, 11)]


def _puzzles_to_motifs(puzzles: list[Puzzle]) -> list[Motif]:
    return [motif for puzzle in puzzles for backbone in puzzle.backbones for motif in backbone.motifs]


def _collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    length_max = max([sample[0].shape[0] for sample in batch])
    x = torch.stack([torch.nn.functional.pad(sample[0], (0, 0, 0, length_max - sample[0].shape[0])) for sample in batch])
    y = torch.stack([sample[1] for sample in batch])
    return x, y


class Data:
    def __init__(self, strategy: Strategy = Strategy.BY_PUZZLE, device: str = "cpu", train_size: float = 0.6,
                 valid_size: float = 0.2, test_size: float = 0.2, random_seed: int = 42) -> None:
        if test_size + valid_size + train_size != 1:
            raise ValueError("Sum of train_size, valid_size, and test_size must equal 1.0")
        puzzles = _read_puzzles()

        self.strategy = strategy
        if self.strategy == Strategy.RANDOM:
            n_motifs = len(motifs := _puzzles_to_motifs(puzzles))
            self.test_size = int(n_motifs * test_size)
            self.valid_size = int(n_motifs * valid_size)
            self.train_size = n_motifs - self.test_size - self.valid_size

            random.seed(random_seed)
            random.shuffle(motifs)

            self.train = Dataset(motifs[:self.train_size], device=device, random_seed=random_seed)
            self.valid = Dataset(motifs[self.train_size:self.train_size + self.valid_size], device=device,
                                 random_seed=random_seed)
            self.test = Dataset(motifs[self.train_size + self.valid_size:], device=device, random_seed=random_seed)
        elif self.strategy == Strategy.BY_PUZZLE:
            n_puzzles = len(puzzles)
            self.test_size = max(int(n_puzzles * test_size), 1)
            self.valid_size = max(int(n_puzzles * valid_size), 1)
            self.train_size = n_puzzles - self.test_size - self.valid_size

            random.seed(random_seed)
            random.shuffle(puzzles)

            self.train = Dataset(_puzzles_to_motifs(puzzles[:self.train_size]), device=device, random_seed=random_seed)
            self.valid = Dataset(_puzzles_to_motifs(puzzles[self.train_size:self.train_size + self.valid_size]), device=device,
                                 random_seed=random_seed)
            self.test = Dataset(_puzzles_to_motifs(puzzles[self.train_size + self.valid_size:]), device=device,
                                random_seed=random_seed)

    def train_loader(self, batch_size: int = 32, shuffle: bool = True,
                     dataloader_kwargs: dict = {}) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn,
                                           **dataloader_kwargs)

    def valid_loader(self, batch_size: int = 32, shuffle: bool = False,
                     dataloader_kwargs: dict = {}) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.valid, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn,
                                           **dataloader_kwargs)

    def test_loader(self, batch_size: int = 32, shuffle: bool = False,
                    dataloader_kwargs: dict = {}) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.test, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn,
                                           **dataloader_kwargs)
