from pathlib import Path

import pandas as pd

from structs import FShape


class FShapeFileReader:
    def __init__(self, file_path: Path) -> None:
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)

    def read(self) -> FShape:
        df = pd.read_csv(
            self.file_path,
            sep="\t",
            na_values="NA",
            names=["fshape", "acid"]
        )
        return FShape.from_dataframe(df)


class FShapeDirectoryReader:
    def __init__(self, directory_path: Path) -> None:
        self.directory_path = Path(directory_path)

        if not self.directory_path.is_dir():
            raise FileNotFoundError(self.directory_path)

    def read(self) -> list[FShape]:
        fshape_files = self.directory_path.glob("*.txt")
        return [FShapeFileReader(file).read() for file in fshape_files]
