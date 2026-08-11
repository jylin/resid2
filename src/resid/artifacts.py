"""Parquet artifact output."""

from pathlib import Path

import pandas as pd

from resid.regression import ResidualizationResult


class ParquetArtifactWriter:
    """Write the same artifacts in a compact, typed columnar format."""

    def write(self, result: ResidualizationResult, output_dir: Path) -> pd.DataFrame:
        destination = output_dir.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "epsilon": result.specific_returns,
            "r": result.returns,
            "X": result.exposures,
            "f": result.factor_returns,
        }
        rows: list[dict[str, object]] = []
        for notation, frame in artifacts.items():
            path = destination / f"{notation}.parquet"
            temporary = destination / f".{notation}.parquet.tmp"
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
            rows.append({"artifact": notation, "rows": len(frame), "path": str(path)})
        return pd.DataFrame(rows)
