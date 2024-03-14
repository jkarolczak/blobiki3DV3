import pandas as pd


class FShape:
    def __init__(self, fshape: list[float], acid: list[str]) -> None:
        self.fshape = fshape
        self.acid = acid

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "FShape":
        return FShape(
            df["fshape"].tolist(),
            df["acid"].tolist()
        )

    def __len__(self):
        return len(self.fshape)

    def __repr__(self):
        return f"FShape(fshape={self.fshape}, acid={self.acid})"