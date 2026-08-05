"""Daily cross-sectional regression implementations."""

from dataclasses import dataclass as plain_dataclass
from dataclasses import field
from typing import Protocol, cast

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


# The dataclasses below hold Series and DataFrames, whose element-wise `==` cannot
# collapse to a single bool, so generated equality would raise instead of compare.
@plain_dataclass(frozen=True, eq=False)
class CrossSectionFit:
    returns: pd.Series
    exposures: pd.DataFrame
    regression_weights: pd.Series
    factor_returns: pd.Series
    fitted_returns: pd.Series
    specific_returns: pd.Series
    rank: int
    r_squared: float
    exposure_centers: pd.Series
    exposure_scales: pd.Series
    # The canonical fit contains only rows usable by every modeled factor.  A
    # stateful factor may still need the full day's observed return vector when
    # it advances its state, so the pipeline attaches it only for that update.
    observed_returns: pd.Series | None = None


@plain_dataclass(frozen=True, eq=False)
class _NormalizedExposures:
    """Normalized exposures with the affine map that produced them."""

    exposures: pd.DataFrame
    centers: pd.Series
    scales: pd.Series


@plain_dataclass(frozen=True, eq=False)
class ResidualizationResult:
    specific_returns: pd.DataFrame
    returns: pd.DataFrame
    exposures: pd.DataFrame
    factor_returns: pd.DataFrame
    diagnostics: pd.DataFrame
    regression_weights: pd.DataFrame
    universe: pd.Series
    model_columns: tuple[str, ...]
    validation_results: tuple[RegressionValidationResult, ...] = ()


class Residualizer(Protocol):
    def fit(
        self,
        returns: pd.Series,
        exposures: pd.DataFrame,
        regression_weights: pd.Series,
    ) -> CrossSectionFit | None: ...


class IncrementalResidualizer(Residualizer, Protocol):
    """A residualizer whose coefficients are a known linear map of the returns.

    `IncrementalRegression` uses that map to revise a fit in place when one return
    changes, so only residualizers that can expose it support live updates.
    """

    def coefficient_projection(
        self,
        exposures: pd.DataFrame,
        regression_weights: pd.Series,
    ) -> np.ndarray:
        """Rows of d(coefficient)/d(return) for already-normalized exposures."""
        ...


@dataclass(frozen=True, slots=True)
class OLSResidualizer:
    winsor_quantile: float = Field(ge=0, lt=0.5)
    unscaled_factors: tuple[str, ...] = ()

    def fit(
        self,
        returns: pd.Series,
        exposures: pd.DataFrame,
        regression_weights: pd.Series,
    ) -> CrossSectionFit | None:
        factor_names = tuple(str(name) for name in exposures.columns)
        _validate_factor_names(factor_names, self.unscaled_factors, "factor names")
        model_columns = ("INTERCEPT", *factor_names)
        cross_section = pd.concat(
            [
                returns.rename("return"),
                exposures,
                regression_weights.rename("regression_weight"),
            ],
            axis=1,
        ).dropna()
        cross_section = _finite_rows(cross_section)
        cross_section = cross_section.loc[cross_section["regression_weight"] > 0]
        if len(cross_section) <= len(model_columns):
            return None

        weights = cross_section["regression_weight"].astype("float64")
        normalized = _normalize(
            cross_section.loc[:, factor_names],
            self.winsor_quantile,
            weights,
            unscaled_factors=self.unscaled_factors,
        )
        if normalized is None:
            return None

        exposures_used = normalized.exposures
        target = cross_section.loc[exposures_used.index, "return"].astype("float64")
        weights = weights.loc[exposures_used.index]
        weight_values = weights.to_numpy(dtype="float64")
        design = _design(exposures_used)
        sqrt_weights = np.sqrt(weight_values)
        coefficients, _, rank, _ = np.linalg.lstsq(
            design * sqrt_weights[:, None],
            target.to_numpy() * sqrt_weights,
            rcond=None,
        )
        if int(rank) < len(model_columns):
            return None
        return _cross_section_fit(
            target=target,
            normalized=normalized,
            regression_weights=weights,
            coefficients=coefficients,
            model_columns=model_columns,
            rank=int(rank),
        )

    def coefficient_projection(
        self,
        exposures: pd.DataFrame,
        regression_weights: pd.Series,
    ) -> np.ndarray:
        weights = _positive_weights(regression_weights, exposures.index)
        sqrt_weights = np.sqrt(weights)
        weighted_design = _design(exposures) * sqrt_weights[:, None]
        # pinv(X sqrt(W)) sqrt(W) is the coefficient map for WLS.
        return np.linalg.pinv(weighted_design) * sqrt_weights


@dataclass(frozen=True, slots=True)
class SequentialOLSResidualizer:
    """Remove explicitly ordered factors through successive univariate OLS fits."""

    factor_order: tuple[str, ...]
    winsor_quantile: float = Field(ge=0, lt=0.5)
    unscaled_factors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_factor_order(self.factor_order, self.unscaled_factors)

    def fit(
        self,
        returns: pd.Series,
        exposures: pd.DataFrame,
        regression_weights: pd.Series,
    ) -> CrossSectionFit | None:
        weights = _uniform_weights(regression_weights, returns.index)
        return _fit_sequential(
            returns=returns,
            exposures=exposures,
            regression_weights=weights,
            factor_order=self.factor_order,
            winsor_quantile=self.winsor_quantile,
            unscaled_factors=self.unscaled_factors,
        )

    def coefficient_projection(
        self,
        exposures: pd.DataFrame,
        regression_weights: pd.Series,
    ) -> np.ndarray:
        weights = _uniform_weights(regression_weights, exposures.index)
        return _sequential_projection(
            self.factor_order, exposures, weights.to_numpy(dtype="float64")
        )


@dataclass(frozen=True, slots=True)
class SequentialWLSResidualizer:
    """Remove ordered factors using positive cross-sectional weights."""

    factor_order: tuple[str, ...]
    winsor_quantile: float = Field(ge=0, lt=0.5)
    unscaled_factors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_factor_order(self.factor_order, self.unscaled_factors)

    def fit(
        self,
        returns: pd.Series,
        exposures: pd.DataFrame,
        regression_weights: pd.Series,
    ) -> CrossSectionFit | None:
        return _fit_sequential(
            returns=returns,
            exposures=exposures,
            regression_weights=regression_weights,
            factor_order=self.factor_order,
            winsor_quantile=self.winsor_quantile,
            unscaled_factors=self.unscaled_factors,
        )

    def coefficient_projection(
        self,
        exposures: pd.DataFrame,
        regression_weights: pd.Series,
    ) -> np.ndarray:
        weights = regression_weights.reindex(exposures.index).to_numpy(dtype="float64")
        return _sequential_projection(self.factor_order, exposures, weights)


def _fit_sequential(
    *,
    returns: pd.Series,
    exposures: pd.DataFrame,
    regression_weights: pd.Series,
    factor_order: tuple[str, ...],
    winsor_quantile: float,
    unscaled_factors: tuple[str, ...] = (),
) -> CrossSectionFit | None:
    exposure_names = tuple(str(name) for name in exposures.columns)
    if set(exposure_names) != set(factor_order) or len(exposure_names) != len(
        factor_order
    ):
        raise ValueError(
            "exposures must match factor_order exactly: "
            f"expected {factor_order}, received {exposure_names}"
        )

    model_columns = ("INTERCEPT", *factor_order)
    ordered_exposures = exposures.loc[:, factor_order]
    cross_section = pd.concat(
        [
            returns.rename("return"),
            ordered_exposures,
            regression_weights.rename("regression_weight"),
        ],
        axis=1,
    ).dropna()
    cross_section = cross_section.loc[cross_section["regression_weight"] > 0]
    cross_section = _finite_rows(cross_section)
    if len(cross_section) <= len(model_columns):
        return None

    weights = cross_section["regression_weight"].astype("float64")
    normalized = _normalize(
        cross_section.loc[:, factor_order],
        winsor_quantile,
        weights,
        unscaled_factors=unscaled_factors,
    )
    if normalized is None:
        return None

    orthogonalized = _orthogonalize(
        normalized.exposures,
        factor_order,
        weights.to_numpy(dtype="float64"),
    )
    if orthogonalized is None:
        return None
    normalized = _NormalizedExposures(
        exposures=orthogonalized,
        centers=normalized.centers,
        scales=normalized.scales,
    )

    exposures_used = normalized.exposures
    target = cross_section.loc[exposures_used.index, "return"].astype("float64")
    weights = weights.loc[exposures_used.index]
    weight_values = weights.to_numpy(dtype="float64")
    remaining = target.to_numpy(dtype="float64", copy=True)
    intercept = float(np.average(remaining, weights=weight_values))
    remaining -= intercept
    coefficients = [intercept]
    for name in factor_order:
        exposure = exposures_used[name].to_numpy(dtype="float64")
        weighted_exposure = weight_values * exposure
        coefficient = float(
            np.dot(weighted_exposure, remaining) / np.dot(weighted_exposure, exposure)
        )
        coefficients.append(coefficient)
        remaining -= exposure * coefficient

    design = _design(exposures_used)
    return _cross_section_fit(
        target=target,
        normalized=normalized,
        regression_weights=weights,
        coefficients=np.asarray(coefficients),
        model_columns=model_columns,
        rank=int(np.linalg.matrix_rank(design)),
    )


@plain_dataclass(slots=True, eq=False)
class IncrementalRegression:
    """Update a provisional linear fit while its cross-section is fixed."""

    residualizer: IncrementalResidualizer
    raw_exposures: pd.DataFrame
    raw_regression_weights: pd.Series
    initial_returns: pd.Series | None = None
    _returns: pd.Series = field(init=False, repr=False)
    _fit: CrossSectionFit | None = field(init=False, default=None, repr=False)
    _projection: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        # A unique exposure index is what lets `update` treat a ticker's index
        # position as a single integer column of the projection.
        if self.raw_exposures.index.has_duplicates:
            raise ValueError("exposure index must contain unique tickers")
        if self.raw_exposures.columns.has_duplicates:
            raise ValueError("exposure names must be unique")
        if self.raw_regression_weights.index.has_duplicates:
            raise ValueError("regression weight index must contain unique tickers")
        self.raw_exposures = self.raw_exposures.copy()
        self.raw_regression_weights = self.raw_regression_weights.reindex(
            self.raw_exposures.index
        ).copy()
        self._returns = pd.Series(
            np.nan,
            index=self.raw_exposures.index,
            dtype="float64",
            name="return",
        )
        if self.initial_returns is not None:
            unknown = self.initial_returns.index.difference(self.raw_exposures.index)
            if len(unknown):
                raise ValueError(f"returns contain unknown tickers: {unknown.tolist()}")
            self._returns.loc[self.initial_returns.index] = self.initial_returns
        self._rebuild()

    @property
    def current_fit(self) -> CrossSectionFit | None:
        return self._fit

    @property
    def returns(self) -> pd.Series:
        return self._returns.copy()

    def update(self, ticker: str, value: float) -> CrossSectionFit | None:
        """Apply one return change and produce a provisional fit when estimable."""

        if ticker not in self._returns.index:
            raise KeyError(f"ticker is not in the period universe: {ticker}")
        if not np.isfinite(value):
            raise ValueError("return update must be finite")

        previous = float(self._returns.loc[ticker])
        self._returns.loc[ticker] = value
        if (
            self._fit is None
            or self._projection is None
            or ticker not in self._fit.returns.index
            or not np.isfinite(previous)
        ):
            return self._rebuild()

        position = cast(int, self._fit.returns.index.get_loc(ticker))
        target = self._fit.returns.copy()
        target.iloc[position] = value
        coefficients = self._fit.factor_returns.to_numpy(dtype="float64", copy=True)
        coefficients += self._projection[:, position] * (value - previous)
        self._fit = _cross_section_fit(
            target=target,
            normalized=_NormalizedExposures(
                exposures=self._fit.exposures,
                centers=self._fit.exposure_centers,
                scales=self._fit.exposure_scales,
            ),
            regression_weights=self._fit.regression_weights,
            coefficients=coefficients,
            model_columns=tuple(str(name) for name in self._fit.factor_returns.index),
            rank=self._fit.rank,
        )
        return self._fit

    def finalize(self) -> CrossSectionFit | None:
        """Reconcile provisional state through the configured canonical fit."""

        return self._rebuild()

    def _rebuild(self) -> CrossSectionFit | None:
        self._fit = self.residualizer.fit(
            self._returns,
            self.raw_exposures,
            self.raw_regression_weights,
        )
        if self._fit is None:
            self._projection = None
            return None
        self._projection = self.residualizer.coefficient_projection(
            self._fit.exposures,
            self._fit.regression_weights,
        )
        return self._fit


def _design(exposures: pd.DataFrame) -> np.ndarray:
    """Exposures prefixed with the intercept column, matching `model_columns`."""

    return np.column_stack(
        [np.ones(len(exposures)), exposures.to_numpy(dtype="float64")]
    )


def _cross_section_fit(
    *,
    target: pd.Series,
    normalized: _NormalizedExposures,
    regression_weights: pd.Series,
    coefficients: np.ndarray,
    model_columns: tuple[str, ...],
    rank: int,
) -> CrossSectionFit:
    exposures = normalized.exposures
    fitted = pd.Series(
        _design(exposures) @ coefficients,
        index=target.index,
        name="fitted_return",
    )
    specific = (target - fitted).rename("specific_return")
    weights = regression_weights.reindex(target.index).astype("float64")
    weighted_mean = float(np.average(target, weights=weights))
    total_ss = float(np.dot(weights, np.square(target - weighted_mean)))
    residual_ss = float(np.dot(weights, np.square(specific)))
    return CrossSectionFit(
        returns=target,
        exposures=exposures,
        regression_weights=weights.rename("regression_weight"),
        factor_returns=pd.Series(
            coefficients,
            index=model_columns,
            name="factor_return",
        ),
        fitted_returns=fitted,
        specific_returns=specific,
        rank=rank,
        r_squared=(1 - residual_ss / total_ss) if total_ss else np.nan,
        exposure_centers=normalized.centers,
        exposure_scales=normalized.scales,
    )


def _finite_rows(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype="float64")
    return frame.loc[np.isfinite(values).all(axis=1)]


def _normalize(
    exposures: pd.DataFrame,
    quantile: float,
    regression_weights: pd.Series | None = None,
    unscaled_factors: tuple[str, ...] = (),
) -> _NormalizedExposures | None:
    """Winsorize, weighted-center, and optionally weighted-scale each exposure."""

    values = exposures.to_numpy(dtype="float64", copy=True)
    lower, upper = np.quantile(values, (quantile, 1 - quantile), axis=0)
    np.clip(values, lower, upper, out=values)
    if regression_weights is None:
        means = values.mean(axis=0)
        dispersion = values.std(axis=0, ddof=0)
    else:
        weights = regression_weights.reindex(exposures.index).to_numpy(dtype="float64")
        means = np.average(values, weights=weights, axis=0)
        dispersion = np.sqrt(
            np.average(np.square(values - means), weights=weights, axis=0)
        )
    # Every column must carry dispersion, including one exempt from scaling: once
    # centered, a constant column is collinear with the intercept, so the fit is
    # not identified and the sequential loop would divide by zero on it.
    if np.any(~np.isfinite(dispersion) | (dispersion <= 0)):
        return None
    exempt = np.fromiter(
        (str(name) in unscaled_factors for name in exposures.columns),
        dtype="bool",
        count=len(exposures.columns),
    )
    scales = np.where(exempt, 1.0, dispersion)
    values -= means
    values /= scales
    return _NormalizedExposures(
        exposures=pd.DataFrame(
            values,
            index=exposures.index,
            columns=exposures.columns,
            copy=False,
        ),
        centers=pd.Series(means, index=exposures.columns, name="exposure_center"),
        scales=pd.Series(scales, index=exposures.columns, name="exposure_scale"),
    )


def _orthogonalize(
    exposures: pd.DataFrame,
    factor_order: tuple[str, ...],
    weights: np.ndarray,
) -> pd.DataFrame | None:
    """Build a weighted orthogonal basis in the configured factor order.

    The first factor keeps its centered/scaled values.  Each later factor is
    residualized against all earlier basis vectors, making the resulting columns
    mutually orthogonal under the regression weights.  Sequential removal on
    this basis is therefore the joint WLS projection while retaining the order's
    first-claim interpretation.
    """

    basis: list[np.ndarray] = []
    for name in factor_order:
        residual = exposures[name].to_numpy(dtype="float64", copy=True)
        original_denominator = float(np.dot(weights * residual, residual))
        for prior in basis:
            denominator = float(np.dot(weights * prior, prior))
            if not np.isfinite(denominator) or denominator <= 0:
                return None
            residual -= (np.dot(weights * residual, prior) / denominator) * prior
        denominator = float(np.dot(weights * residual, residual))
        tolerance = np.finfo("float64").eps * max(1.0, original_denominator) * 100
        if not np.isfinite(denominator) or denominator <= tolerance:
            return None
        basis.append(residual)
    return pd.DataFrame(
        np.column_stack(basis),
        index=exposures.index,
        columns=factor_order,
    )


def _positive_weights(weights: pd.Series, index: pd.Index) -> np.ndarray:
    aligned = weights.reindex(index)
    values = aligned.to_numpy(dtype="float64")
    if (
        len(values) != len(index)
        or not np.isfinite(values).all()
        or not (values > 0).all()
    ):
        raise ValueError("regression weights must be finite and positive")
    return values


def _uniform_weights(weights: pd.Series, index: pd.Index) -> pd.Series:
    values = _positive_weights(weights, index)
    if len(values) and not np.allclose(values, values[0], rtol=0, atol=1e-12):
        raise ValueError(
            "SequentialOLSResidualizer only accepts uniform regression weights; "
            "use SequentialWLSResidualizer or OLSResidualizer for weighted fits"
        )
    return pd.Series(1.0, index=index, name="regression_weight")


def _sequential_projection(
    factor_order: tuple[str, ...],
    exposures: pd.DataFrame,
    weights: np.ndarray,
) -> np.ndarray:
    """Express each sequential coefficient as a linear functional of the returns.

    Stage k regresses the running residual on its own exposure, so its weight row
    is that exposure's normal-equation row minus the projections onto every row
    already fitted — the intercept first, then each earlier factor.
    """

    intercept_row = weights / weights.sum()
    coefficient_rows = [intercept_row]
    prior_exposures: list[np.ndarray] = []
    for name in factor_order:
        exposure = exposures[name].to_numpy(dtype="float64")
        weighted_exposure = weights * exposure
        denominator = np.dot(weighted_exposure, exposure)
        coefficient_row = weighted_exposure / denominator
        coefficient_row -= weighted_exposure.sum() / denominator * intercept_row
        for prior_exposure, prior_row in zip(
            prior_exposures, coefficient_rows[1:], strict=True
        ):
            coefficient_row -= (
                np.dot(weighted_exposure, prior_exposure) / denominator * prior_row
            )
        coefficient_rows.append(coefficient_row)
        prior_exposures.append(exposure)
    return np.vstack(coefficient_rows)


def _validate_factor_order(
    factor_order: tuple[str, ...],
    unscaled_factors: tuple[str, ...] = (),
) -> None:
    _validate_factor_names(factor_order, unscaled_factors, "factor_order")


def _validate_factor_names(
    factor_names: tuple[str, ...],
    unscaled_factors: tuple[str, ...],
    label: str,
) -> None:
    if not factor_names:
        raise ValueError(f"{label} must contain at least one factor")
    if any(not name or name == "INTERCEPT" for name in factor_names):
        raise ValueError("factor names must be non-empty and cannot be INTERCEPT")
    if len(set(factor_names)) != len(factor_names):
        raise ValueError(f"{label} must contain unique names")
    unknown = tuple(name for name in unscaled_factors if name not in factor_names)
    if unknown:
        raise ValueError(f"unscaled_factors are not modeled factors: {unknown}")
