"""Validation boundary and core residualization checks."""

from typing import Protocol

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic.dataclasses import dataclass

from resid.regression import RegressionValidationResult, ResidualizationResult


class RegressionValidationError(ValueError):
    def __init__(self, failures: tuple[RegressionValidationResult, ...]):
        self.failures = failures
        details = "; ".join(f"{item.name}: {item.message}" for item in failures)
        super().__init__(f"residualization validation failed: {details}")


class RegressionValidation(Protocol):
    def validate(self, result: ResidualizationResult) -> RegressionValidationResult: ...


@dataclass(frozen=True, slots=True)
class RegressionCoverageValidation:
    minimum_date_coverage: float = Field(ge=0, le=1)

    def validate(self, result: ResidualizationResult) -> RegressionValidationResult:
        expected_dates = (
            result.universe.loc[result.universe].index.get_level_values("date").unique()
        )
        completed_dates = pd.DatetimeIndex(result.factor_returns["date"].unique())
        coverage = (
            len(completed_dates) / len(expected_dates) if len(expected_dates) else 0.0
        )
        passed = bool(coverage >= self.minimum_date_coverage)
        return RegressionValidationResult(
            name="regression_coverage",
            passed=passed,
            message=(
                f"completed {len(completed_dates)} of {len(expected_dates)} dates "
                f"({coverage:.2%}; required {self.minimum_date_coverage:.2%})"
            ),
            metrics={
                "completed_dates": len(completed_dates),
                "expected_dates": len(expected_dates),
                "date_coverage": coverage,
            },
        )


@dataclass(frozen=True, slots=True)
class FiniteOutputValidation:
    def validate(self, result: ResidualizationResult) -> RegressionValidationResult:
        frames = (
            result.specific_returns,
            result.returns,
            result.exposures,
            result.factor_returns,
            result.diagnostics,
            result.regression_weights,
        )
        nonfinite = sum(
            int(
                np.count_nonzero(
                    ~np.isfinite(
                        frame.select_dtypes(include="number").to_numpy(dtype="float64")
                    )
                )
            )
            for frame in frames
        )
        return RegressionValidationResult(
            name="finite_outputs",
            passed=nonfinite == 0,
            message=f"found {nonfinite} non-finite numeric outputs",
            metrics={"nonfinite_outputs": nonfinite},
        )


@dataclass(frozen=True, slots=True)
class ReturnReconstructionValidation:
    absolute_tolerance: float = Field(gt=0)

    def validate(self, result: ResidualizationResult) -> RegressionValidationResult:
        terms = list(result.model_columns)
        exposures = result.exposures.set_index(["date", "ticker"])[terms]
        dates = exposures.index.get_level_values("date")
        factor_returns = result.factor_returns.set_index("date").loc[dates, terms]
        fitted = np.einsum(
            "ij,ij->i",
            exposures.to_numpy(dtype="float64"),
            factor_returns.to_numpy(dtype="float64"),
        )
        observed = (
            result.returns.set_index(["date", "ticker"])["return"]
            .reindex(exposures.index)
            .to_numpy(dtype="float64")
        )
        specific = (
            result.specific_returns.set_index(["date", "ticker"])["specific_return"]
            .reindex(exposures.index)
            .to_numpy(dtype="float64")
        )
        errors = observed - fitted - specific
        maximum_error = float(np.abs(errors).max()) if len(errors) else np.inf
        passed = bool(
            np.isfinite(maximum_error) and maximum_error <= self.absolute_tolerance
        )
        return RegressionValidationResult(
            name="return_reconstruction",
            passed=passed,
            message=(
                f"maximum error {maximum_error:.3g} "
                f"(tolerance {self.absolute_tolerance:.3g})"
            ),
            metrics={"maximum_reconstruction_error": maximum_error},
        )


@dataclass(frozen=True, slots=True)
class ResidualOrthogonalityValidation:
    absolute_tolerance: float = Field(gt=0)

    def validate(self, result: ResidualizationResult) -> RegressionValidationResult:
        terms = list(result.model_columns)
        exposures = result.exposures.set_index(["date", "ticker"])[terms]
        specific = result.specific_returns.set_index(["date", "ticker"])[
            "specific_return"
        ].reindex(exposures.index)
        weights = result.regression_weights.set_index(["date", "ticker"])[
            "regression_weight"
        ].reindex(exposures.index)
        weight_sums = weights.groupby(level="date").sum()
        moments = (
            exposures.mul(specific * weights, axis=0)
            .groupby(level="date")
            .sum()
            .div(weight_sums, axis=0)
            .abs()
        )
        maximum_error = float(moments.to_numpy().max())
        passed = bool(
            np.isfinite(maximum_error) and maximum_error <= self.absolute_tolerance
        )
        return RegressionValidationResult(
            name="residual_orthogonality",
            passed=passed,
            message=(
                f"maximum exposure-residual moment {maximum_error:.3g} "
                f"(tolerance {self.absolute_tolerance:.3g})"
            ),
            metrics={"maximum_absolute_factor_moment": maximum_error},
        )


@dataclass(frozen=True, slots=True)
class SequentialOrthogonalityValidation:
    """Check orthogonality immediately after each ordered factor is removed."""

    absolute_tolerance: float = Field(gt=0)

    def validate(self, result: ResidualizationResult) -> RegressionValidationResult:
        if len(result.model_columns) < 2 or result.model_columns[0] != "INTERCEPT":
            return RegressionValidationResult(
                name="sequential_orthogonality",
                passed=False,
                message="sequential results require an intercept and ordered factors",
                metrics={"maximum_absolute_stage_moment": np.inf},
            )

        factor_order = result.model_columns[1:]
        expected_columns = result.model_columns

        index_columns = ["date", "ticker"]
        exposures = result.exposures.set_index(index_columns)[list(factor_order)]
        dates = exposures.index.get_level_values("date")
        factor_returns = result.factor_returns.set_index("date").loc[
            dates, list(expected_columns)
        ]
        remaining = (
            result.returns.set_index(index_columns)["return"]
            .reindex(exposures.index)
            .to_numpy(dtype="float64", copy=True)
        )
        remaining -= factor_returns["INTERCEPT"].to_numpy(dtype="float64")
        weights = (
            result.regression_weights.set_index(index_columns)["regression_weight"]
            .reindex(exposures.index)
            .to_numpy(dtype="float64")
        )
        weight_series = pd.Series(weights, index=exposures.index)
        weight_sums = weight_series.groupby(level="date").sum()

        maximum_error = float(
            pd.Series(remaining * weights, index=exposures.index)
            .groupby(level="date")
            .sum()
            .div(weight_sums)
            .abs()
            .max()
        )
        for name in factor_order:
            values = exposures[name].to_numpy(dtype="float64")
            remaining -= values * factor_returns[name].to_numpy(dtype="float64")
            moments = (
                pd.Series(remaining * values * weights, index=exposures.index)
                .groupby(level="date")
                .sum()
                .div(weight_sums)
                .abs()
            )
            maximum_error = max(maximum_error, float(moments.max()))

        passed = bool(
            np.isfinite(maximum_error) and maximum_error <= self.absolute_tolerance
        )
        return RegressionValidationResult(
            name="sequential_orthogonality",
            passed=passed,
            message=(
                f"maximum stage exposure-residual moment {maximum_error:.3g} "
                f"(tolerance {self.absolute_tolerance:.3g})"
            ),
            metrics={"maximum_absolute_stage_moment": maximum_error},
        )
