"""Composable cross-sectional return residualization."""

from resid.artifacts import CsvArtifactWriter
from resid.data import (
    AnalysisWindow,
    FixedTopMarketCapUniverse,
    MarcapDataSource,
    MarketDataSource,
    UniverseBuilder,
    analysis_window,
)
from resid.factors import (
    CharacteristicFactorModel,
    Factor,
    FactorModel,
    PreparedFactorModel,
    momentum_factor,
    size_factor,
)
from resid.market_beta import RecursiveMarketBetaModel
from resid.pipeline import run_pipeline
from resid.regression import (
    CrossSectionFit,
    OLSResidualizer,
    RegressionValidationResult,
    ResidualizationResult,
    Residualizer,
)
from resid.returns import PercentageReturns, ReturnCalculator
from resid.validation import (
    FiniteOutputValidation,
    RegressionCoverageValidation,
    RegressionValidation,
    RegressionValidationError,
    ReturnReconstructionValidation,
)
