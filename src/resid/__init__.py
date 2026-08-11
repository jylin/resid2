"""Composable cross-sectional return residualization."""

from resid.artifacts import ParquetArtifactWriter
from resid.data import (
    AnalysisWindow,
    DailyTopMarketCapUniverse,
    FixedTopMarketCapUniverse,
    MarcapDataSource,
    MarketDataSource,
    UniverseBuilder,
    analysis_window,
    history_start,
    previous_session_values,
    universe_dates,
    universe_index,
    universe_members,
)
from resid.events import (
    EventLog,
    JsonlEventLog,
    LiveEvent,
    LoggedEvent,
    PeriodClosed,
    PeriodKey,
    PeriodOpened,
    ReturnObserved,
)
from resid.factors import (
    CharacteristicFactorModel,
    Factor,
    FactorModel,
    PreparedFactorModel,
    long_term_reversal_factor,
    momentum_factor,
    size_factor,
)
from resid.live import LivePeriodResult, LiveRegressionRunner, historical_events
from resid.market_beta import RecursiveMarketBetaModel
from resid.pipeline import run_pipeline
from resid.regression import (
    CrossSectionFit,
    IncrementalRegression,
    IncrementalResidualizer,
    OLSResidualizer,
    RegressionValidationResult,
    ResidualizationResult,
    Residualizer,
    SequentialOLSResidualizer,
    SequentialWLSResidualizer,
)
from resid.returns import PercentageReturns, ReturnCalculator
from resid.validation import (
    FiniteOutputValidation,
    RegressionCoverageValidation,
    RegressionValidation,
    RegressionValidationError,
    ResidualOrthogonalityValidation,
    ReturnReconstructionValidation,
    SequentialOrthogonalityValidation,
)
from resid.weights import (
    EqualRegressionWeights,
    RegressionWeightModel,
    SquareRootMarketCapWeights,
)
