"""Ganzen Ordner auswerten -> eine Summary-CSV (+ optional PDFs/JSON je Bild).

Auch ohne GUI nutzbar:
    python -m beamshaper.batch "C:/Messungen/2026-09-21" --pixel 1.45 --mag 43.75 
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .analysis import AnalysisSettings, analyze_file
from .export import (output_dir, save_figures, save_report_for, save_results, today,
                     write_summary_csv)
from .report import compact
from .imageio import list_images


def run_batch(folder: str | Path, settings: AnalysisSettings, save_pdf: bool = True,
              save_json: bool = True, progress: Callable[[int, int, str], bool] | None = None,
              files: list[Path] | None = None, plot_opts: dict | None = None,
              save_report: bool = True,
              lattice_files: set | None = None) -> tuple[Path, list[dict]]:
    """Wertet alle Bilder aus. `progress(i, n, name)` -> False bricht ab.

    Ergebnis: Summary-CSV, je Bild JSON + PDF-Plots, und ein PDF-Report mit
    Vergleichsseite und einer Seite je Bild.

    lattice_files: Dateinamen, fuer die die Uniformity-/Crosstalk-Abschaetzung
    im Fallengitter gerechnet wird (None = Einstellung settings.lattice_estimate
    fuer alle).
    """
    files = files if files is not None else list_images(folder)
    rows = []
    for_report = []
    for i, p in enumerate(files):
        if progress and progress(i, len(files), p.name) is False:
            break
        try:
            s_i = settings
            if lattice_files is not None:
                s_i = replace(settings, lattice_estimate=p.name in lattice_files)
            res = analyze_file(p, s_i)
            rows.append(res.summary())
            if save_json:
                save_results(res)
            if save_pdf:
                save_figures(res, **(plot_opts or {}))
            if save_report:
                for_report.append(compact(res))
        except Exception as exc:  # ein kaputtes Bild soll den Batch nicht stoppen
            rows.append({"file": p.name, "warnings": f"FEHLER: {exc}"})
    out = output_dir(Path(folder) / "x") / f"{today()}_batch_summary.csv"
    write_summary_csv(rows, out)
    if save_report and for_report:
        if progress:
            progress(len(files), len(files), "Report ...")
        save_report_for(for_report, Path(folder) / "x", "batch", plot_opts)
    if progress:
        progress(len(files), len(files), "fertig")
    return out, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="FlatTop-Batch-Auswertung")
    ap.add_argument("folder")
    ap.add_argument("--pixel", type=float, default=1.45, help="Kamerapixel in um (Standard IDS/Sony)")
    ap.add_argument("--mag", type=float, default=43.75, help="Abbildungsmassstab (Standard 40er Objektiv)")
    ap.add_argument("--background", default=None, help="Hintergrundbild")
    ap.add_argument("--no-pdf", action="store_true", help="keine PDF-Plots speichern")
    ap.add_argument("--pitch", type=float, default=5.288, help="Pitch des Fallengitters in um")
    ap.add_argument("--no-lattice", nargs="*", default=None, metavar="DATEI",
                    help="ohne Dateinamen: keine Crosstalk-Abschaetzung; mit Dateinamen: "
                         "fuer diese Dateien nicht")
    ap.add_argument("--no-regions", action="store_true",
                    help="Ringe 1/e2, 50 %%, 90 %% in keinem Plot und im Report einzeichnen")
    a = ap.parse_args(argv)
    s = replace(AnalysisSettings(), pixel_size_um=a.pixel, magnification=a.mag,
                background_path=a.background, pitch_um=a.pitch)
    lattice_files = None
    if a.no_lattice is not None:
        if len(a.no_lattice) == 0:
            s = replace(s, lattice_estimate=False)
        else:
            skip = {Path(x).name for x in a.no_lattice}
            lattice_files = {p.name for p in list_images(a.folder)} - skip
    out, rows = run_batch(a.folder, s, save_pdf=not a.no_pdf,
                          plot_opts={"show_regions": not a.no_regions},
                          lattice_files=lattice_files,
                          progress=lambda i, n, name: print(f"[{i}/{n}] {name}"))
    print(f"{len(rows)} Bilder -> {out}")


if __name__ == "__main__":
    main()
