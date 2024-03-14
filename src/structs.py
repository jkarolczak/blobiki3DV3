from __future__ import annotations

import numpy as np
import pandas as pd


class FShape:
    def __init__(self, fshape: np.ndarray, acid: np.ndarray) -> None:
        self.fshape = fshape
        self.acid = acid
        self.meta = dict()

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> FShape:
        return FShape(
            df["fshape"].to_numpy(),
            df["acid"].to_numpy()
        )

    def __len__(self):
        return len(self.fshape)

    def __repr__(self):
        return f"FShape(fshape={self.fshape.tolist()}, acid={self.acid.tolist()})"
