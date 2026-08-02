from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from resid import (
    AnalysisWindow,
    CharacteristicFactorModel,
    CrossSectionFit,
    CsvArtifactWriter,
    DailyTopMarketCapUniverse,
    EqualRegressionWeights,
    Factor,
    FiniteOutputValidation,
    FixedTopMarketCapUniverse,
    MarcapDataSource,
    OLSResidualizer,
    PercentageReturns,
    RegressionCoverageValidation,
    RegressionValidationError,
    ReturnReconstructionValidation,
    SequentialOLSResidualizer,
    SequentialOrthogonalityValidation,
    SequentialWLSResidualizer,
    SquareRootMarketCapWeights,
    analysis_window,
    long_term_reversal_factor,
    momentum_factor,
    run_pipeline,
    size_factor,
)


@dataclass
class FrameSource:
    frame: pd.DataFrame

    def latest_date(self, end: str | None = None) -> pd.Timestamp:
        dates = self.frame.index.get_level_values("date")
        if end:
            dates = dates[dates <= pd.Timestamp(end)]
        return cast(pd.Timestamp, pd.Timestamp(dates.max()))

    def trading_dates(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        dates = self.frame.index.get_level_values("date")
        return pd.DatetimeIndex(dates[(dates >= start) & (dates <= end)].unique())

    def load(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: pd.Index | None = None,
    ) -> pd.DataFrame:
        dates = self.frame.index.get_level_values("date")
        selected = self.frame.loc[(dates >= start) & (dates <= end)]
        if tickers is not None:
            selected = selected.loc[
                selected.index.get_level_values("ticker").isin(tickers)
            ]
        return selected


def canonical_frame(
    dates: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return pd.DataFrame(
        {
            "return_percent": np.tile(np.linspace(-2, 2, len(tickers)), len(dates)),
            "market_cap": np.tile(np.arange(1, len(tickers) + 1), len(dates)),
        },
        index=index,
    )


def test_fixed_universe_uses_only_first_output_date() -> None:
    dates = pd.date_range("2025-01-02", periods=2, freq="B")
    frame = canonical_frame(dates, ["A", "B", "C", "D"])
    frame.loc[(dates[1], "A"), "market_cap"] = 1_000
    source = FrameSource(frame)

    universe = FixedTopMarketCapUniverse(size=3).build(
        source, AnalysisWindow(dates[0], dates[-1])
    )

    assert set(universe.xs(dates[0]).index) == {"B", "C", "D"}
    assert set(universe.xs(dates[1]).index) == {"B", "C", "D"}


def test_daily_universe_uses_previous_day_market_cap() -> None:
    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    frame = canonical_frame(dates, ["A", "B", "C"])
    frame.loc[(dates[0], slice(None)), "market_cap"] = [1, 3, 2]
    frame.loc[(dates[1], slice(None)), "market_cap"] = [5, 1, 4]
    frame.loc[(dates[2], "B"), "market_cap"] = 1_000

    universe = DailyTopMarketCapUniverse(size=2).build(
        FrameSource(frame), AnalysisWindow(dates[1], dates[2])
    )

    assert set(universe.xs(dates[1]).index) == {"B", "C"}
    assert set(universe.xs(dates[2]).index) == {"A", "C"}


def test_return_calculation_converts_percentage_to_decimal() -> None:
    frame = canonical_frame(pd.date_range("2025-01-02", periods=1), ["A", "B"])

    returns = PercentageReturns().calculate(frame)

    np.testing.assert_allclose(returns, [-0.02, 0.02])


def test_square_root_market_cap_weights_use_the_prior_session() -> None:
    dates = pd.date_range("2025-01-02", periods=3, freq="B")
    frame = canonical_frame(dates, ["A", "B"])
    frame.loc[(dates[0], slice(None)), "market_cap"] = [100, 400]
    frame.loc[(dates[1], slice(None)), "market_cap"] = [900, 1_600]

    weights = SquareRootMarketCapWeights().calculate(frame)

    assert weights.loc[(dates[0], "A")] != weights.loc[(dates[0], "A")]
    np.testing.assert_allclose(weights.xs(dates[1]), [10.0, 20.0])
    np.testing.assert_allclose(weights.xs(dates[2]), [30.0, 40.0])


def test_factor_exposures_do_not_use_current_return() -> None:
    dates = pd.date_range("2024-12-20", periods=7, freq="B")
    frame = canonical_frame(dates, ["A", "B"])
    calculator = PercentageReturns()
    model = CharacteristicFactorModel(
        (
            size_factor(name="SIZE"),
            momentum_factor(
                lookback_days=3,
                skip_days=1,
                min_periods=3,
                name="MOMENTUM",
            ),
        )
    )
    baseline = model.exposures(frame, calculator.calculate(frame))
    changed = frame.copy()
    changed.loc[(dates[4], "A"), "return_percent"] = 90.0
    perturbed = model.exposures(changed, calculator.calculate(changed))

    pd.testing.assert_series_equal(
        baseline.loc[(dates[4], "A")], perturbed.loc[(dates[4], "A")]
    )


def test_size_and_long_term_reversal_directions() -> None:
    dates = pd.date_range("2025-01-02", periods=6, freq="B")
    returns = pd.DataFrame(
        {"SMALL": [0.01] * len(dates), "BIG": [-0.01] * len(dates)},
        index=dates,
    )
    market_caps = pd.DataFrame(
        {"SMALL": [1e8] * len(dates), "BIG": [1e10] * len(dates)},
        index=dates,
    )

    size = size_factor(name="SIZE").build(returns, market_caps)
    hml_factor = long_term_reversal_factor(
        lookback_days=3,
        skip_days=1,
        min_periods=3,
        name="HML",
    )
    hml = hml_factor.build(returns, market_caps)

    assert hml_factor.history_business_days == 4
    assert size.loc[dates[-1], "SMALL"] < size.loc[dates[-1], "BIG"]
    assert hml.loc[dates[-1], "SMALL"] < hml.loc[dates[-1], "BIG"]


def test_ols_recovers_cross_sectional_returns() -> None:
    tickers = [f"A{i}" for i in range(8)]
    index = pd.Index(tickers, name="ticker")
    size = np.linspace(-2, 2, len(tickers))
    momentum = np.sin(np.linspace(0, 2 * np.pi, len(tickers)))
    returns = pd.Series(
        0.01 + 0.02 * size - 0.03 * momentum, index=index, name="return"
    )
    exposures = pd.DataFrame({"SIZE": size, "MOMENTUM": momentum}, index=index)

    fit = OLSResidualizer(winsor_quantile=0).fit(
        returns, exposures, pd.Series(1.0, index=index)
    )

    assert fit is not None
    assert fit.factor_returns.index.tolist() == ["INTERCEPT", "SIZE", "MOMENTUM"]
    assert fit.specific_returns.abs().max() < 1e-12
    assert fit.rank == 3


def test_regressions_drop_nonfinite_rows() -> None:
    index = pd.Index([f"A{i}" for i in range(5)], name="ticker")
    returns = pd.Series([0.01, np.inf, 0.02, 0.03, 0.04], index=index)
    exposures = pd.DataFrame({"SIZE": np.arange(5.0)}, index=index)
    weights = pd.Series(1.0, index=index)

    for residualizer in (
        OLSResidualizer(winsor_quantile=0),
        SequentialWLSResidualizer(factor_order=("SIZE",), winsor_quantile=0),
    ):
        fit = residualizer.fit(returns, exposures, weights)
        assert fit is not None
        assert len(fit.returns) == 4
        assert np.isfinite(fit.factor_returns).all()
        assert np.isfinite(fit.specific_returns).all()


def test_ols_rejects_empty_and_duplicate_factor_names() -> None:
    index = pd.Index([f"A{i}" for i in range(4)], name="ticker")
    returns = pd.Series(np.arange(4.0), index=index)
    weights = pd.Series(1.0, index=index)

    with pytest.raises(ValueError, match="at least one factor"):
        OLSResidualizer(winsor_quantile=0).fit(
            returns, pd.DataFrame(index=index), weights
        )

    duplicate_exposures = pd.DataFrame(
        np.column_stack([np.arange(4.0), np.arange(4.0)]),
        index=index,
        columns=["SIZE", "SIZE"],
    )
    with pytest.raises(ValueError, match="unique"):
        OLSResidualizer(winsor_quantile=0).fit(returns, duplicate_exposures, weights)


def test_winsorization_and_standardization_match_pandas() -> None:
    index = pd.Index([f"A{i}" for i in range(8)], name="ticker")
    exposures = pd.DataFrame(
        {
            "SIZE": [-20.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 30.0],
            "MOMENTUM": [10.0, 4.0, 2.0, 1.0, -1.0, -2.0, -4.0, -12.0],
        },
        index=index,
    )
    returns = pd.Series(np.linspace(-0.04, 0.03, len(index)), index=index)
    quantile = 0.125

    fit = OLSResidualizer(winsor_quantile=quantile).fit(
        returns, exposures, pd.Series(1.0, index=index)
    )

    assert fit is not None
    clipped = exposures.clip(
        lower=exposures.quantile(quantile),
        upper=exposures.quantile(1 - quantile),
        axis="columns",
    )
    expected = (clipped - clipped.mean()) / clipped.std(ddof=0)
    np.testing.assert_allclose(fit.exposures, expected, atol=1e-15)


def test_sequential_ols_removes_factors_in_explicit_order() -> None:
    tickers = pd.Index([f"A{i}" for i in range(10)], name="ticker")
    market = np.linspace(-2, 2, len(tickers))
    size = 0.8 * market + np.cos(np.linspace(0, 2 * np.pi, len(tickers)))
    momentum = -0.4 * size + np.sin(np.linspace(0, 3 * np.pi, len(tickers)))
    exposures = pd.DataFrame(
        {"SIZE": size, "MOMENTUM": momentum, "MARKET_BETA": market},
        index=tickers,
    )
    returns = pd.Series(
        0.01 + 0.03 * market - 0.02 * size + 0.015 * momentum,
        index=tickers,
        name="return",
    )
    order = ("MARKET_BETA", "SIZE", "MOMENTUM")

    equal_weights = pd.Series(1.0, index=tickers)
    fit = SequentialOLSResidualizer(factor_order=order, winsor_quantile=0).fit(
        returns, exposures, equal_weights
    )

    assert fit is not None
    assert fit.factor_returns.index.tolist() == ["INTERCEPT", *order]
    assert fit.exposures.columns.tolist() == list(order)
    remaining = fit.returns.to_numpy() - fit.factor_returns["INTERCEPT"]
    for name in order:
        exposure = fit.exposures[name].to_numpy()
        expected = np.dot(exposure, remaining) / np.dot(exposure, exposure)
        assert fit.factor_returns[name] == pytest.approx(expected)
        remaining -= exposure * expected
        assert np.mean(remaining * exposure) == pytest.approx(0, abs=1e-15)
    np.testing.assert_allclose(fit.specific_returns, remaining, atol=1e-15)

    reversed_fit = SequentialOLSResidualizer(
        factor_order=tuple(reversed(order)), winsor_quantile=0
    ).fit(returns, exposures, equal_weights)
    assert reversed_fit is not None
    assert not np.allclose(fit.specific_returns, reversed_fit.specific_returns)
    assert fit.rank == 4


def test_sequential_wls_uses_weighted_moments() -> None:
    tickers = pd.Index([f"A{i}" for i in range(8)], name="ticker")
    exposures = pd.DataFrame(
        {
            "FIRST": np.linspace(-2, 2, len(tickers)),
            "SECOND": np.cos(np.linspace(0, 2 * np.pi, len(tickers))),
        },
        index=tickers,
    )
    returns = pd.Series(np.linspace(-0.03, 0.05, len(tickers)), index=tickers)
    weights = pd.Series(np.geomspace(1, 16, len(tickers)), index=tickers)
    order = ("FIRST", "SECOND")

    fit = SequentialWLSResidualizer(factor_order=order, winsor_quantile=0).fit(
        returns, exposures, weights
    )

    assert fit is not None
    remaining = fit.returns.to_numpy() - fit.factor_returns["INTERCEPT"]
    weight_values = fit.regression_weights.to_numpy()
    assert np.average(remaining, weights=weight_values) == pytest.approx(0)
    for name in order:
        exposure = fit.exposures[name].to_numpy()
        expected = np.dot(weight_values * exposure, remaining) / np.dot(
            weight_values * exposure, exposure
        )
        assert fit.factor_returns[name] == pytest.approx(expected)
        remaining -= exposure * expected
        assert np.average(remaining * exposure, weights=weight_values) == pytest.approx(
            0, abs=1e-15
        )
    np.testing.assert_allclose(fit.specific_returns, remaining, atol=1e-15)


def test_unscaled_factor_keeps_its_raw_dispersion() -> None:
    tickers = pd.Index([f"A{i}" for i in range(10)], name="ticker")
    beta = np.linspace(0.5, 1.7, len(tickers))
    exposures = pd.DataFrame(
        {"BETA": beta, "SIZE": np.linspace(24.0, 30.0, len(tickers))},
        index=tickers,
    )
    returns = pd.Series(np.linspace(-0.02, 0.03, len(tickers)), index=tickers)
    weights = pd.Series(np.geomspace(1, 32, len(tickers)), index=tickers)
    order = ("BETA", "SIZE")

    fit = SequentialWLSResidualizer(
        factor_order=order, winsor_quantile=0, unscaled_factors=("BETA",)
    ).fit(returns, exposures, weights)

    assert fit is not None
    weight_values = fit.regression_weights.to_numpy()

    def weighted_std(values: np.ndarray) -> float:
        mean = np.average(values, weights=weight_values)
        return float(
            np.sqrt(np.average(np.square(values - mean), weights=weight_values))
        )

    assert fit.exposure_scales["BETA"] == 1.0
    assert fit.exposure_scales["SIZE"] == pytest.approx(weighted_std(exposures["SIZE"]))
    assert weighted_std(fit.exposures["BETA"].to_numpy()) == pytest.approx(
        weighted_std(beta)
    )
    assert weighted_std(fit.exposures["SIZE"].to_numpy()) == pytest.approx(1.0)
    for name in order:
        column = fit.exposures[name].to_numpy()
        assert np.average(column, weights=weight_values) == pytest.approx(0, abs=1e-14)
        restored = column * fit.exposure_scales[name] + fit.exposure_centers[name]
        np.testing.assert_allclose(restored, exposures[name], atol=1e-12)


def test_pydantic_checks_component_bounds() -> None:
    with pytest.raises(ValidationError):
        FixedTopMarketCapUniverse(size=0)
    with pytest.raises(ValidationError):
        DailyTopMarketCapUniverse(size=0)
    with pytest.raises(ValidationError):
        OLSResidualizer(winsor_quantile=0.5)
    with pytest.raises(ValidationError):
        SequentialOLSResidualizer(factor_order=(), winsor_quantile=0)
    with pytest.raises(ValidationError):
        SequentialOLSResidualizer(factor_order=("SIZE", "SIZE"), winsor_quantile=0)
    with pytest.raises(ValidationError):
        SequentialWLSResidualizer(factor_order=(), winsor_quantile=0)
    with pytest.raises(ValidationError):
        SequentialWLSResidualizer(
            factor_order=("SIZE",), winsor_quantile=0, unscaled_factors=("MKT",)
        )
    with pytest.raises(ValidationError):
        RegressionCoverageValidation(minimum_date_coverage=1.1)


def test_factor_model_rejects_empty_or_duplicate_factors() -> None:
    with pytest.raises(ValueError, match="at least one factor"):
        CharacteristicFactorModel(())
    with pytest.raises(ValueError, match="must be unique"):
        CharacteristicFactorModel((size_factor(name="SIZE"), size_factor(name="SIZE")))


def test_full_marcap_pipeline_exports_csv(tmp_path: Path) -> None:
    data_dir = tmp_path / "marcap" / "data"
    data_dir.mkdir(parents=True)
    dates = pd.date_range("2024-01-02", "2025-06-30", freq="B")
    tickers = [f"{ticker:06d}" for ticker in range(6)]
    rows = [
        {
            "Date": day,
            "Code": ticker,
            "ChangesRatio": 0.2 * np.sin(day_number / 11 + ticker_number)
            + 0.05 * ticker_number,
            "Marcap": (ticker_number + 1)
            * 1_000_000
            * (1 + 0.0002 * day_number * (ticker_number + 1)),
        }
        for day_number, day in enumerate(dates)
        for ticker_number, ticker in enumerate(tickers)
    ]
    data = pd.DataFrame(rows)
    for year, year_frame in data.groupby(data["Date"].dt.year):
        year_frame.to_parquet(data_dir / f"marcap-{year}.parquet", index=False)

    source = MarcapDataSource(data_dir)
    window = analysis_window(source, years=1, end=None)
    factor_model = CharacteristicFactorModel(
        (
            size_factor(name="SIZE"),
            momentum_factor(
                lookback_days=20,
                skip_days=5,
                min_periods=15,
                name="MOMENTUM",
            ),
        )
    )
    result = run_pipeline(
        window=window,
        source=source,
        universe_builder=FixedTopMarketCapUniverse(size=5),
        return_calculator=PercentageReturns(),
        factor_model=factor_model,
        residualizer=SequentialOLSResidualizer(
            factor_order=("SIZE", "MOMENTUM"), winsor_quantile=0.01
        ),
        regression_weight_model=EqualRegressionWeights(),
    )
    output_dir = tmp_path / "output"
    manifest = CsvArtifactWriter().write(result, output_dir)

    first_members = result.universe.xs(pd.Timestamp("2024-07-01")).index
    assert tuple(first_members) == tuple(reversed(tickers[1:]))
    assert result.factor_returns["date"].max() == dates[-1]
    assert result.exposures.columns.tolist() == [
        "date",
        "ticker",
        "INTERCEPT",
        "SIZE",
        "MOMENTUM",
    ]
    assert manifest["artifact"].tolist() == ["epsilon", "r", "X", "f"]
    assert {path.name for path in output_dir.iterdir()} == {
        "epsilon.csv",
        "r.csv",
        "X.csv",
        "f.csv",
    }


def test_pipeline_components_are_replaceable() -> None:
    dates = pd.date_range("2025-01-02", periods=8, freq="B")
    source = FrameSource(canonical_frame(dates, [f"A{i}" for i in range(6)]))
    factor_calls = 0

    def reversal(returns: pd.DataFrame, _: pd.DataFrame) -> pd.DataFrame:
        nonlocal factor_calls
        factor_calls += 1
        return -returns.shift(1)

    @dataclass
    class CountingResidualizer:
        calls: int = 0

        def fit(
            self,
            returns: pd.Series,
            exposures: pd.DataFrame,
            regression_weights: pd.Series,
        ) -> CrossSectionFit | None:
            self.calls += 1
            return OLSResidualizer(winsor_quantile=0.01).fit(
                returns, exposures, regression_weights
            )

    residualizer = CountingResidualizer()

    result = run_pipeline(
        window=AnalysisWindow(dates[0], dates[-1]),
        source=source,
        universe_builder=FixedTopMarketCapUniverse(size=5),
        return_calculator=PercentageReturns(),
        factor_model=CharacteristicFactorModel(
            (Factor("REVERSAL", reversal, history_business_days=1),)
        ),
        residualizer=residualizer,
        regression_weight_model=EqualRegressionWeights(),
    )

    assert factor_calls == 1
    assert residualizer.calls == len(dates)
    assert result.model_columns == ("INTERCEPT", "REVERSAL")


def test_pipeline_runs_caller_specified_validations() -> None:
    dates = pd.date_range("2025-01-02", periods=8, freq="B")
    source = FrameSource(canonical_frame(dates, [f"A{i}" for i in range(6)]))

    def reversal(returns: pd.DataFrame, _: pd.DataFrame) -> pd.DataFrame:
        return -returns.shift(1)

    def run(minimum_coverage: float):
        return run_pipeline(
            window=AnalysisWindow(dates[0], dates[-1]),
            source=source,
            universe_builder=FixedTopMarketCapUniverse(size=5),
            return_calculator=PercentageReturns(),
            factor_model=CharacteristicFactorModel(
                (Factor("REVERSAL", reversal, history_business_days=1),)
            ),
            residualizer=SequentialOLSResidualizer(
                factor_order=("REVERSAL",), winsor_quantile=0.01
            ),
            regression_weight_model=EqualRegressionWeights(),
            validations=(
                RegressionCoverageValidation(minimum_date_coverage=minimum_coverage),
                FiniteOutputValidation(),
                ReturnReconstructionValidation(absolute_tolerance=1e-12),
                SequentialOrthogonalityValidation(absolute_tolerance=1e-12),
            ),
        )

    result = run(0.8)

    assert [item.name for item in result.validation_results] == [
        "regression_coverage",
        "finite_outputs",
        "return_reconstruction",
        "sequential_orthogonality",
    ]
    assert all(item.passed for item in result.validation_results)
    with pytest.raises(RegressionValidationError, match="regression_coverage"):
        run(1.0)
