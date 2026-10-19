"""Ergebnisse speichern.

Ablage (neben den Messbildern):
    <Bildordner>/Auswertung/Plots/JJJJ-MM-TT/<name>_<figur>.pdf   Plots tageweise
    <Bildordner>/Auswertung/JJJJ-MM-TT_<name>_results.json          Kennzahlen + Einstellungen
    <Bildordner>/Auswertung/JJJJ-MM-TT_<name>_flatness.csv          Flatness-Kurve
    <Bildordner>/Auswertung/JJJJ-MM-TT_batch_summary.csv            Batch-Tabelle
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .analysis import FlatTopResult
from .plotting import (FIGURES, EXPORT_SIZES, DEFAULT_EXPORT, TITLE_H_MM, TITLE_W_MM,
                       _rc, title_figsize, comparison_order, plot_comparison)


def today() -> str:
    return dt.date.today().isoformat()


def output_dir(image_path: str | Path, base: str | Path | None = None) -> Path:
    root = Path(base) if base else Path(image_path).parent / "Auswertung"
    root.mkdir(parents=True, exist_ok=True)
    return root


def plot_dir(image_path: str | Path, base: str | Path | None = None) -> Path:
    d = output_dir(image_path, base) / "Plots" / today()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean(v):
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if math.isnan(v) else v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, tuple):
        return [_clean(x) for x in v]
    return v


def _deep_clean(v):
    if isinstance(v, dict):
        return {k: _deep_clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_deep_clean(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return _clean(v)


def result_to_dict(res: FlatTopResult) -> dict:
    f = res.fit
    s = res.um_per_px
    return {
        "file": res.path,
        "analysed": dt.datetime.now().isoformat(timespec="seconds"),
        "settings": {k: _clean(v) for k, v in asdict(res.settings).items()},
        "fit_supergauss_px": {
            "A": f.A, "x0": f.x0, "y0": f.y0, "wa": f.wa, "wb": f.wb,
            "theta_rad": f.theta, "m": f.m, "B": f.B, "binning": f.binning,
            "errors": f.errors, "R2": f.r_squared,
        },
        "radii_um": {k: {"long": a, "short": b} for k, (a, b) in res.radii_um().items()},
        "summary": {k: _clean(v) for k, v in res.summary().items()},
        "lattice_estimate": _deep_clean(res.lattice),
        "simulation": _deep_clean(res.simulation),
        "warnings": res.warnings,
    }


def save_results(res: FlatTopResult, base: str | Path | None = None) -> list[Path]:
    out = output_dir(res.path, base)
    stem = f"{today()}_{res.name}"
    p_json = out / f"{stem}_results.json"
    p_json.write_text(json.dumps(result_to_dict(res), indent=2, default=_clean), encoding="utf-8")
    p_csv = out / f"{stem}_flatness.csv"
    np.savetxt(p_csv, np.column_stack([res.flatness_d_um, res.flatness_u]),
               delimiter=",", header="d_um,uniformity_std_over_mean", comments="")
    return [p_json, p_csv]


def save_figures(res: FlatTopResult, which=None, base: str | Path | None = None,
                 **kw) -> list[Path]:
    """Speichert jede Figur als eigenes PDF, ohne Titel (Parameter stehen im JSON)."""
    fmt = "pdf"
    which = which or list(DEFAULT_EXPORT)
    out = plot_dir(res.path, base)
    paths = []
    for key in which:
        _, func = FIGURES[key]
        if key == "title":    # exakt so gross wie das Fenster auf der Titelseite
            size = title_figsize(kw.get("title_w_mm", TITLE_W_MM),
                                 kw.get("title_h_mm", TITLE_H_MM))
        else:
            size = EXPORT_SIZES[key]
        fig = Figure(figsize=size)
        extra = {k: v for k, v in kw.items() if k in func.__code__.co_varnames}
        if key == "title":
            extra["show_logo_hint"] = False      # Logo-Kasten nur in der Vorschau
        func(fig, res, **extra)
        p = out / f"{res.name}_{key}.{fmt}"
        # kein bbox_inches="tight": die Breite soll exakt 16 cm bleiben
        with _rc():                       # Schrift-Einbettung (Type 42) greift beim Speichern
            fig.savefig(p, dpi=300)
        paths.append(p)
    return paths


def save_comparison(res_a: FlatTopResult, res_b: FlatTopResult,
                    base: str | Path | None = None, **kw) -> Path:
    """Vergleich zweier Profile (Simulation links) als PDF, 16 cm breit.

    Ablage neben der Messung (bzw. dem rechten Bild):
    Auswertung/Plots/JJJJ-MM-TT/<rechts>_vs_<links>_comparison.pdf
    """
    (left, _), (right, _) = comparison_order(res_a, res_b)
    fig = Figure(figsize=EXPORT_SIZES["comparison"])
    extra = {k: v for k, v in kw.items() if k in plot_comparison.__code__.co_varnames}
    plot_comparison(fig, res_a, res_b, **extra)
    p = plot_dir(right.path, base) / f"{right.name}_vs_{left.name}_comparison.pdf"
    with _rc():
        fig.savefig(p, dpi=300)
    return p


def save_report_for(results, base_image: str | Path, name: str, plot_opts=None,
                    base: str | Path | None = None) -> Path:
    """PDF-Report in <Bildordner>/Auswertung/JJJJ-MM-TT_<name>_report.pdf."""
    from .report import save_report
    out = output_dir(base_image, base) / f"{today()}_{name}_report.pdf"
    return save_report(list(results), out, plot_opts or {})


def write_summary_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in keys})
    return path


def _fmt(v):
    v = _clean(v)
    if isinstance(v, float):
        return f"{v:.6g}".replace(".", ",")   # deutsches Excel
    return "" if v is None else v
