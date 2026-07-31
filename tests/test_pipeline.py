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
    Factor,
    FiniteOutputValidation,
    FixedTopMarketCapUniverse,
    MarcapDataSource,
    OLSResidualizer,
    PercentageReturns,
    RegressionCoverageValidation,
    RegressionValidationError,
    ReturnReconstructionValidation,
    analysis_window,
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


def test_return_calculation_converts_percentage_to_decimal() -> None:
    frame = canonical_frame(pd.date_range("2025-01-02", periods=1), ["A", "B"])

    returns = PercentageReturns().calculate(frame)

    np.testing.assert_allclose(returns, [-0.02, 0.02])


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


def test_ols_recovers_cross_sectional_returns() -> None:
    tickers = [f"A{i}" for i in range(8)]
    index = pd.Index(tickers, name="ticker")
    size = np.linspace(-2, 2, len(tickers))
    momentum = np.sin(np.linspace(0, 2 * np.pi, len(tickers)))
    returns = pd.Series(
        0.01 + 0.02 * size - 0.03 * momentum, index=index, name="return"
    )
    exposures = pd.DataFrame({"SIZE": size, "MOMENTUM": momentum}, index=index)

    fit = OLSResidualizer(winsor_quantile=0).fit(returns, exposures)

    assert fit is not None
    assert fit.factor_returns.index.tolist() == ["INTERCEPT", "SIZE", "MOMENTUM"]
    assert fit.specific_returns.abs().max() < 1e-12
    assert fit.rank == 3


def test_pydantic_checks_component_bounds() -> None:
    with pytest.raises(ValidationError):
        FixedTopMarketCapUniverse(size=0)
    with pytest.raises(ValidationError):
        OLSResidualizer(winsor_quantile=0.5)
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
        residualizer=OLSResidualizer(winsor_quantile=0.01),
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
        ) -> CrossSectionFit | None:
            self.calls += 1
            return OLSResidualizer(winsor_quantile=0.01).fit(returns, exposures)

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
            residualizer=OLSResidualizer(winsor_quantile=0.01),
            validations=(
                RegressionCoverageValidation(minimum_date_coverage=minimum_coverage),
                FiniteOutputValidation(),
                ReturnReconstructionValidation(absolute_tolerance=1e-12),
            ),
        )

    result = run(0.8)

    assert [item.name for item in result.validation_results] == [
        "regression_coverage",
        "finite_outputs",
        "return_reconstruction",
    ]
    assert all(item.passed for item in result.validation_results)
    with pytest.raises(RegressionValidationError, match="regression_coverage"):
        run(1.0)
