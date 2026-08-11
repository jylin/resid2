from io import StringIO

from resid.progress import TerminalProgress, reported_stage


def test_terminal_progress_reports_colored_stage_and_inner_progress() -> None:
    stream = StringIO()
    progress = TerminalProgress(
        total_stages=1,
        stream=stream,
        min_interval=0,
        use_color=True,
    )

    with reported_stage(
        progress, "Residualize returns", total=4, unit="dates"
    ) as stage:
        stage.update(2)
        stage.update(4)

    output = stream.getvalue()
    assert "[1/1]" in output
    assert "Residualize returns" in output
    assert "50%" in output
    assert "4/4 dates" in output
    assert "dates/s" in output
    assert "done in" in output
    assert "\033[" in output


def test_terminal_progress_uses_plain_output_for_redirected_streams() -> None:
    stream = StringIO()
    progress = TerminalProgress(total_stages=1, stream=stream, min_interval=0)

    with reported_stage(progress, "Load data"):
        pass

    output = stream.getvalue()
    assert "[1/1] Load data" in output
    assert output.count("[1/1]") == 1
    assert "done in" in output
    assert "\033[" not in output


def test_terminal_progress_prints_summary_below_the_progress_line() -> None:
    stream = StringIO()
    progress = TerminalProgress(total_stages=1, stream=stream, min_interval=0)

    with reported_stage(progress, "Prepare factors", total=2) as stage:
        stage.update(1, "momentum")
        stage.update(2)
        stage.summary("tables:\n  exposures: 100 rows x 3 columns [SIZE,HML,MOM]")

    lines = stream.getvalue().splitlines()
    progress_lines = [line for line in lines if "[1/1]" in line]
    assert progress_lines
    assert all("exposures:" not in line for line in progress_lines)
    assert any(line.startswith("  status: done in") for line in lines)
    assert "  details:" in lines
    assert "    tables:" in lines
    assert "      exposures: 100 rows x 3 columns [SIZE,HML,MOM]" in lines
