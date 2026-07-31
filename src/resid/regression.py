"""Daily cross-sectional regression boundary and OLS implementation."""

from dataclasses import dataclass as plain_dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic.dataclasses import dataclass


@plain_dataclass(frozen=True)
class RegressionValidationResult:
    name: str
    passed: bool
    message: str
    metrics: dict[str, float | int]


@plain_dataclass(frozen=True)
class CrossSectionFit:
    returns: pd.Series
    exposures: pd.DataFrame
    factor_returns: pd.Series
    fitted_returns: pd.Series
    specific_returns: pd.Series
    rank: int
    r_squared: float


@plain_dataclass(frozen=True)
class ResidualizationResult:
    specific_returns: pd.DataFrame
    returns: pd.DataFrame
    exposures: pd.DataFrame
    factor_returns: pd.DataFrame
    diagnostics: pd.DataFrame
    universe: pd.Series
    model_columns: tuple[str, ...]
    validation_results: tuple[RegressionValidationResult, ...] = ()


class Residualizer(Protocol):
    def fit(
        self,
        returns: pd.Series,
        exposures: pd.DataFrame,
    ) -> CrossSectionFit | None: ...


@dataclass(frozen=True, slots=True)
class OLSResidualizer:
    winsor_quantile: float = Field(ge=0, lt=0.5)

    def fit(
        self,
        returns: pd.Series,
        exposures: pd.DataFrame,
    ) -> CrossSectionFit | None:
        factor_names = tuple(str(name) for name in exposures.columns)
        model_columns = ("INTERCEPT", *factor_names)
        cross_section = pd.concat(
            [returns.rename("return"), exposures], axis=1
        ).dropna()
        if len(cross_section) <= len(model_columns):
            return None

        normalized = _normalize(
            cross_section.loc[:, factor_names], self.winsor_quantile
        )
        if normalized is None:
            return None

        target = cross_section.loc[normalized.index, "return"].astype("float64")
        design = np.column_stack([np.ones(len(normalized)), normalized])
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, target.to_numpy(), rcond=None
        )
        fitted = pd.Series(
            design @ coefficients,
            index=target.index,
            name="fitted_return",
        )
        specific = (target - fitted).rename("specific_return")
        total_ss = float(np.square(target - target.mean()).sum())
        residual_ss = float(np.square(specific).sum())

        return CrossSectionFit(
            returns=target,
            exposures=normalized,
            factor_returns=pd.Series(
                coefficients,
                index=model_columns,
                name="factor_return",
            ),
            fitted_returns=fitted,
            specific_returns=specific,
            rank=int(rank),
            r_squared=1 - residual_ss / total_ss if total_ss else np.nan,
        )


def _normalize(exposures: pd.DataFrame, quantile: float) -> pd.DataFrame | None:
    normalized = pd.DataFrame(index=exposures.index)
    for name in exposures:
        values = exposures[name].clip(
            exposures[name].quantile(quantile),
            exposures[name].quantile(1 - quantile),
        )
        scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            return None
        normalized[name] = (values - values.mean()) / scale
    return normalized
