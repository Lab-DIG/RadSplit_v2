"""Command line interface: ``radsplit split``, ``audit``, ``holdout`` and ``scan``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .api import split as make_split
from .audit import audit as run_audit

app = typer.Typer(
    help="Patient disjoint, covariate balanced data partitioning and partition audits.",
    no_args_is_help=True,
    add_completion=False,
)


def _read(path: Path, labels: Optional[Path] = None, on: str = "patient_id") -> pd.DataFrame:
    if path.is_dir():
        from radsplit import manifest

        dataset = manifest.scan(path)
        if labels is not None:
            dataset = dataset.join(pd.read_csv(labels), on=on)
        typer.echo(f"Scanned {len(dataset.table)} files below {path}")
        return dataset.series()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _ratios(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in text.split(","):
        name, value = item.split("=")
        out[name.strip()] = float(value)
    return out


@app.command("split")
def split_command(
    table: Path = typer.Argument(
        ..., exists=True, readable=True,
        help="CSV or Parquet table, or a folder of DICOM, NIfTI, MetaImage or NRRD files"),
    patient: str = typer.Option("patient_id", help="Patient identifier column"),
    target: Optional[list[str]] = typer.Option(None, help="Label column (repeatable)"),
    balance: Optional[list[str]] = typer.Option(None, help="Column to balance (repeatable)"),
    ratios: str = typer.Option("train=0.7,tune=0.1,test=0.2", help="name=fraction pairs"),
    folds: Optional[int] = typer.Option(None, help="Create this many cross validation folds"),
    fixed: Optional[str] = typer.Option(None, help="Column with assignments to keep"),
    labels: Optional[Path] = typer.Option(
        None, help="CSV with labels to join when TABLE is a folder"),
    labels_on: str = typer.Option("patient_id", help="Key column shared with the labels CSV"),
    seed: int = typer.Option(0, help="Random seed"),
    output: Path = typer.Option(Path("radsplit"), help="Output directory"),
) -> None:
    """Create a patient disjoint, covariate balanced partition."""
    result = make_split(
        _read(table, labels, labels_on), patient=patient, target=target, balance=balance,
        ratios=None if folds else _ratios(ratios), folds=folds, fixed=fixed, seed=seed,
    )
    result.save(output)
    typer.echo(result.report())
    typer.echo(f"\nWritten to {output}")


@app.command("audit")
def audit_command(
    table: Path = typer.Argument(..., exists=True, readable=True, help="CSV or Parquet table"),
    patient: str = typer.Option(..., help="Patient identifier column"),
    split: str = typer.Option("split", help="Column with the partition of each record"),
    variable: list[str] = typer.Option(..., help="Column to compare (repeatable)"),
    also_disjoint: Optional[list[str]] = typer.Option(None, help="Other keys to check"),
    output: Optional[Path] = typer.Option(None, help="Write report (.md or .html)"),
    strict: bool = typer.Option(False, help="Exit with status 1 on overlap or imbalance"),
) -> None:
    """Audit an existing partition for patient overlap and covariate imbalance."""
    report = run_audit(
        _read(table), patient=patient, split=split, variables=variable,
        also_disjoint=also_disjoint,
    )
    text = report.to_markdown()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        content = report.to_html() if output.suffix.lower() == ".html" else text
        output.write_text(content, encoding="utf-8")
    typer.echo(text)
    if strict and not (report.disjoint and report.balanced):
        raise typer.Exit(1)


@app.command("holdout")
def holdout_command(
    table: Path = typer.Argument(..., exists=True, readable=True, help="CSV or Parquet table"),
    patient: str = typer.Option("patient_id", help="Patient identifier column"),
    by: str = typer.Option(..., help="Column that defines the held out group, e.g. manufacturer"),
    value: list[str] = typer.Option(..., help="Value to hold out as the test set (repeatable)"),
    balance: Optional[list[str]] = typer.Option(None, help="Column to balance (repeatable)"),
    mixed: str = typer.Option("exclude", help="exclude, test or train (see documentation)"),
    seed: int = typer.Option(0, help="Random seed"),
    output: Path = typer.Option(Path("radsplit"), help="Output directory"),
) -> None:
    """Hold out a scanner, kernel, site or any other group as the test set."""
    from .shift import holdout

    result = holdout(_read(table), patient=patient, by=by, values=value, balance=balance,
                     mixed=mixed, seed=seed)
    result.save(output)
    typer.echo(result.report())
    typer.echo(f"\nWritten to {output}")


@app.command("scan")
def scan_command(
    folder: Path = typer.Argument(..., exists=True, file_okay=False,
                                  help="Folder of DICOM, NIfTI, MetaImage or NRRD files"),
    output: Path = typer.Option(Path("manifest.csv"), help="Series table to write"),
    labels: Optional[Path] = typer.Option(None, help="CSV with labels to join"),
    labels_on: str = typer.Option("patient_id", help="Key column shared with the labels CSV"),
) -> None:
    """Read image headers below a folder into a series table and check integrity."""
    import json

    from radsplit import manifest

    dataset = manifest.scan(folder)
    if labels is not None:
        dataset = dataset.join(pd.read_csv(labels), on=labels_on)
    dataset.series().to_csv(output, index=False)
    report = dataset.validate()
    typer.echo(json.dumps({**dataset.summary(), "valid": report["valid"],
                           "issues": report["counts"]}, indent=2))
    typer.echo(f"\nWritten to {output}")
    if not report["valid"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
