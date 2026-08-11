"""Parquet artifact output."""

from pathlib import Path

import pandas as pd

from resid.progress import StageProgress
from resid.regression import ResidualizationResult


class ParquetArtifactWriter:
    """Write the same artifacts in a compact, typed columnar format."""

    @staticmethod
    def _table_preview(notation: str, frame: pd.DataFrame) -> str:
        key_fields = []
        for name in ("date", "ticker"):
            if name in frame.columns:
                key_fields.append((name, int(frame[name].nunique())))
        values = [
            str(column) for column in frame.columns if column not in {"date", "ticker"}
        ]
        lines = [f"{notation}: {len(frame):,} rows"]
        if key_fields:
            dimensions = " x ".join(f"{count:,} {name}s" for name, count in key_fields)
            lines.append(f"  keys: {dimensions}")
        lines.append(f"  values: {', '.join(values)}")
        return "\n".join(lines)

    def write(
        self,
        result: ResidualizationResult,
        output_dir: Path,
        progress: StageProgress | None = None,
    ) -> pd.DataFrame:
        destination = output_dir.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "epsilon": result.specific_returns,
            "r": result.returns,
            "X": result.exposures,
            "f": result.factor_returns,
        }
        rows: list[dict[str, object]] = []
        previews: list[str] = []
        for position, (notation, frame) in enumerate(artifacts.items(), start=1):
            path = destination / f"{notation}.parquet"
            temporary = destination / f".{notation}.parquet.tmp"
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
            rows.append({"artifact": notation, "rows": len(frame), "path": str(path)})
            previews.append(self._table_preview(notation, frame))
            if progress is not None:
                progress.update(position, f"write {notation}.parquet")
        if progress is not None:
            progress.summary(
                "\n".join(("tables:", *(f"  {preview}" for preview in previews)))
            )
        return pd.DataFrame(rows)
