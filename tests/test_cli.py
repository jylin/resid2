from pathlib import Path

import pytest
from pydantic import ValidationError

from resid.cli import apply_cli_overrides, build_parser, load_config


def test_load_config_reads_nested_toml(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text(
        """
[run]
years = 10

[output]
directory = "out"

[universe]
method = "daily"
size = 500

[factors]
order = ["MKT", "SIZE", "HML", "MOM"]

[factors.momentum]
lookback_days = 189
skip_days = 63
min_periods = 126

[factors.value]
lookback_days = 1008
skip_days = 252
min_periods = 756

[factors.market_beta]
enabled = true
lookback_days = 252
min_periods = 126
decay = 0.97

[regression]
method = "sqrt-cap-wls"
winsor_quantile = 0.01

[validation]
minimum_date_coverage = 0.99
tolerance = 1e-12
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.run.years == 10
    assert config.universe.size == 500
    assert config.factors.order == ("MKT", "SIZE", "HML", "MOM")
    assert config.factors.market_beta.enabled
    assert config.regression.method == "sqrt-cap-wls"


def test_cli_values_override_config(tmp_path: Path) -> None:
    path = tmp_path / "run.toml"
    path.write_text(
        """
[run]
years = 10

[output]
directory = "out"

[universe]
method = "daily"
size = 500

[factors]
order = ["MKT", "SIZE", "HML", "MOM"]

[factors.momentum]
lookback_days = 189
skip_days = 63
min_periods = 126

[factors.value]
lookback_days = 1008
skip_days = 252
min_periods = 756

[factors.market_beta]
enabled = true
lookback_days = 252
min_periods = 126
decay = 0.97

[regression]
method = "ols"
winsor_quantile = 0.01

[validation]
minimum_date_coverage = 0.99
tolerance = 1e-12
""",
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "prices",
            "--config",
            str(path),
            "--years",
            "5",
            "--universe-size",
            "1000",
            "--regression-method",
            "sqrt-cap-wls",
        ]
    )

    config = apply_cli_overrides(load_config(path), args)

    assert args.data_dir == Path("prices")
    assert config.run.years == 5
    assert config.universe.size == 1000
    assert config.regression.method == "sqrt-cap-wls"


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text("[universe]\nnumber = 10\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)
