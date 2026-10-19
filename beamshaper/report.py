"""PDF-Report: je Bild eine A4-Seite mit Uebersichtsplot und Tabellen,
bei mehreren Bildern vorneweg eine Vergleichsseite.

Inhalt je Bild
  - Uebersicht (2D-Bild mit den drei Regionen, horizontaler/vertikaler Schnitt)
  - Radien w der drei Regionen (1/e^2, 50 %, 90 %) entlang der Ellipsen-
    Halbachsen (lang/kurz) und entlang horizontal/vertikal
  - Uniformity sigma/mu und Peak-to-Valley in jeder Region
  - Neigungswinkel der langen Achse, Fitguete, Kantenbreite, Warnungen

Nur matplotlib (PdfPages) - keine weiteren Abhaengigkeiten.
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .analysis import FlatTopResult, level_labels
from . import plotting as pl

A4 = (8.27, 11.69)
HEAD_BG = "#E6E6EE"
GRID = "#B8B8C4"
MUTED = "#55555F"
WARN = "#B3261E"

LEVEL_NAMES = {"1/e2": r"$1/e^2$", "50%": "50 %", "90%": "90 %"}


def _c(v, dec=2):
    """Zahl mit Dezimalkomma; nan -> Strich."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.{dec}f}".replace(".", ",")


def _table(ax, header, rows, col_widths=None, fontsize=9, first_col_left=False):
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center",
                 colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, 1.55)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(HEAD_BG)
            cell.set_text_props(weight="bold")
        if c == 0 and first_col_left and r > 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.04
    return t


def _page_for(res: FlatTopResult, plot_opts: dict) -> Figure:
    s = res.summary()
    labels = level_labels(res.settings)
    fig = Figure(figsize=A4)
    with pl._rc():
        # --- Kopf -----------------------------------------------------------
        fig.text(0.07, 0.955, f"FlatTop-Auswertung: {res.name}", fontsize=15, weight="bold")
        ny, nx = res.image_shape
        fig.text(0.07, 0.935,
                 f"{Path(res.path).name}   ·   {dt.date.today():%d.%m.%Y}   ·   "
                 f"{nx} × {ny} Pixel   ·   {_c(res.um_per_px * 1000, 2)} nm/Pixel "
                 f"(Pixel {_c(res.settings.pixel_size_um, 2)} µm, Vergrößerung "
                 f"{_c(res.settings.magnification, 2)})",
                 fontsize=8.5, color=MUTED)

        # --- Uebersicht -----------------------------------------------------
        gs = fig.add_gridspec(2, 2, left=0.08, right=0.95, top=0.905, bottom=0.555,
                              width_ratios=[1.6, 1], wspace=0.35, hspace=0.45)
        ax_im = fig.add_subplot(gs[:, 0])
        ax_h = fig.add_subplot(gs[0, 1])
        ax_v = fig.add_subplot(gs[1, 1], sharex=ax_h, sharey=ax_h)
        opts = {k: v for k, v in plot_opts.items()
                if k in ("cmap", "show_regions", "show_text", "show_fit", "show_circles",
                         "show_scalebar")}
        pl.draw_overview(ax_im, ax_h, ax_v, res, **opts)

        # --- Tabelle: Radien, Uniformity und Crosstalk je Region ------------
        L = res.lattice
        fig.text(0.07, 0.508, "Radien w, Uniformity und Crosstalk", fontsize=11.5,
                 weight="bold")
        if L:
            at = L["atom"]
            fig.text(0.07, 0.5,
                     f"Crosstalk: gemessenes Profil auf Quadratgitter, Pitch {_c(L['pitch_um'], 3)} µm, "
                     f"{8 if L['neighbour_ring'] == 1 else 'alle'} Nachbarn\nAtom: "
                     f"{_c(at['mass_u'], 1)} u, T = {_c(at['T_uK'], 1)} µK, "
                     f"$\\nu_r$ = {_c(at['nu_kHz'], 1)} kHz → $\\sigma_\\mathrm{{atom}}$ = {_c(L['sigma_atom_um'] * 1e3, 0)} nm"
                     f"   ·   Untergrund {_c(L['background'])} ± {_c(L['background_err'])} Counts",
                     fontsize=7.6, color=MUTED, va="top", linespacing=1.4)
        else:
            fig.text(0.07, 0.492, "Crosstalk-Abschätzung für dieses Bild nicht berechnet.",
                     fontsize=7.6, color=MUTED)
        header = ["Region", "w lang\n(µm)", "w kurz\n(µm)", "w horizontal\n(µm)",
                  "w vertikal\n(µm)", "Uniformity\n$\\sigma/\\mu$ (%)", "Crosstalk\n$\\eta$ (%)",
                  "Peak-Valley\n(%)"]

        def eta_txt(v, e):
            if v is None or not np.isfinite(v):
                return "–"
            return f"{_c(100 * v)} ± {_c(100 * e)}"

        rows = []
        for lab in labels:
            eta = eta_txt(L["hard"][lab]["eta"], L["hard"][lab]["eta_err"]) if L else "–"
            rows.append([LEVEL_NAMES.get(lab, lab) + " (hart)",
                         _c(s[f"w_long_{lab}_um"]), _c(s[f"w_short_{lab}_um"]),
                         _c(s[f"w_h_{lab}_um"]), _c(s[f"w_v_{lab}_um"]),
                         _c(s[f"U_{lab}_pct"]), eta, _c(s[f"PV_{lab}_pct"], 1)])
        if L:
            rows.append(["atomgewichtet", "–", "–", "–", "–", _c(100 * L["U_w"]),
                         eta_txt(L["eta_w"], L["eta_w_err"]), "–"])
        ax_t = fig.add_axes([0.07, 0.305, 0.86, 0.165])
        _table(ax_t, header, rows,
               col_widths=[0.16, 0.1, 0.1, 0.12, 0.11, 0.12, 0.15, 0.12], fontsize=8.6)

        # --- Kenngroessen ---------------------------------------------------
        f = res.fit
        theta = math.degrees(f.theta)
        fig.text(0.07, 0.34, "Ellipse und Fit", fontsize=11.5, weight="bold")
        theta_txt = f"{_c(theta, 1)}°"
        if s["ellipticity_e2"] > 0.97:
            theta_txt += " (fast rund)"
        info = [
            ["Neigungswinkel lange Achse $\\theta$", theta_txt,
             "Elliptizität b/a (1/e²)", _c(s["ellipticity_e2"], 3)],
            ["Supergauss-Ordnung m", f"{_c(f.m, 2)} ± {_c(f.errors.get('m', float('nan')), 2)}",
             "Fitgüte R²", _c(f.r_squared, 4)],
            ["Kante 10–90 % lang / kurz", f"{_c(s['edge_long_um'])} / {_c(s['edge_short_um'])} µm",
             "Zentrum x / y", f"{_c(s['x0_um'], 1)} / {_c(s['y0_um'], 1)} µm"],
            [f"Uniformity < {_c(s['uniformity_threshold_pct'], 0)} % bis Radius",
             (f"{_c(s['r_uniform_um'])} µm (Ø {_c(2 * s['r_uniform_um'])} µm)"
              if np.isfinite(s["r_uniform_um"]) else "nicht erreicht"),
             "Signal-Ø inkl. Nebenstruktur", f"{_c(s['D_signal_um'], 1)} µm"],
            ["Max. Pixelwert / übersteuert",
             f"{res.max_value:.0f} / {_c(s['saturated_pct'], 2)} %",
             "Plateau-Uniformity (90 %)", f"{_c(s['U_plateau_rms_pct'])} %"],
        ]
        ax_i = fig.add_axes([0.07, 0.185, 0.86, 0.145])
        t = ax_i.table(cellText=info, loc="upper center", cellLoc="left",
                       colWidths=[0.28, 0.26, 0.26, 0.2])
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.45)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor(GRID)
            cell.set_linewidth(0.5)
            if c in (0, 2):
                cell.set_facecolor("#F4F4F8")
        ax_i.axis("off")

        # --- Warnungen und Definitionen ------------------------------------
        y = 0.165
        for w in res.warnings:
            fig.text(0.07, y, "Warnung: " + w, fontsize=9, color=WARN)
            y -= 0.018
        notes = (
            "w: Radius (halbe Breite) bis zu dem Punkt, an dem die Intensität auf den Level "
            "(1/e², 50 %, 90 % des Maximums über dem Untergrund) gefallen ist; Durchmesser = 2w.\n"
            "lang/kurz: entlang der Ellipsen-Halbachsen (gefittete Ellipse, Winkel $\\theta$ gegen die "
            "Bild-x-Achse, Bild-y nach unten).\n"
            "horizontal/vertikal: Schnittpunkt der Level-Ellipse mit der Bild-x- bzw. y-Achse "
            "durch das Zentrum.\n"
            "Uniformity $\\sigma/\\mu$ (hart): Standardabweichung/Mittelwert aller Pixel innerhalb der "
            "jeweiligen Level-Ellipse (Untergrund abgezogen).\n"
            "Crosstalk $\\eta$ (hart): Σ I_Nachbarn / Σ I_eigen über der Level-Ellipse; I_Nachbarn = "
            "gemessenes Profil, um ±Pitch auf die 8 Nachbar-Sites verschoben.\n"
            "atomgewichtet: Gewicht W = $\\exp(-r^2/2\\sigma_\\mathrm{atom}^2)$ (thermische Atomverteilung), "
            "$U_w$ = $\\sigma_W/\\mu_W$, $\\eta_w$ = Σ I_Nachbarn·W / Σ I_eigen·W (wie Multitone_Simulation).\n"
            "± bei $\\eta$: Unsicherheit aus dem Untergrund (ungleichmäßiger Rand, $8\\,\\delta B / \\langle I_\\mathrm{eigen}\\rangle$).\n"
            "Uniformity-Radius: größter Kreis um das Zentrum, in dem σ/µ noch unter der "
            "Schwelle liegt (aus der Flatness-Kurve, linear interpoliert).\n"
            "Peak-Valley: (99,5-%-Perzentil − 0,5-%-Perzentil)/Mittelwert, damit einzelne "
            "Staubkörner/Hotpixel den Wert nicht bestimmen.\n"
            r"Radien aus dem 2D-Supergauss-Fit $I = A\,\exp(-(r^2)^m) + B$."
        )
        fig.text(0.07, 0.035, notes, fontsize=7.2, color=MUTED, va="bottom", linespacing=1.45)
    return fig


def _summary_page(results: list[FlatTopResult], folder: str) -> Figure:
    fig = Figure(figsize=A4)
    with pl._rc():
        fig.text(0.07, 0.955, "FlatTop-Auswertung: Vergleich", fontsize=15, weight="bold")
        fig.text(0.07, 0.935, f"{folder}   ·   {dt.date.today():%d.%m.%Y}   ·   "
                              f"{len(results)} Bilder", fontsize=8.5, color=MUTED)
        labels = level_labels(results[0].settings)
        n = len(labels)
        # Tabelle 1: Radien
        thr = _c(results[0].summary()["uniformity_threshold_pct"], 0)
        header = ["Datei", "$\\theta$\n(°)"]
        header += [f"w {LEVEL_NAMES.get(l, l)} (µm)\nlang / kurz" for l in labels]
        header += [f"r mit U < {thr} %\n(µm)"]
        rows = []
        for r in results:
            s = r.summary()
            row = [r.name, _c(math.degrees(r.fit.theta), 1)]
            row += [f"{_c(s[f'w_long_{l}_um'])} / {_c(s[f'w_short_{l}_um'])}" for l in labels]
            row += [_c(s["r_uniform_um"]) if np.isfinite(s["r_uniform_um"]) else "–"]
            rows.append(row)
        h1 = min(0.3, 0.021 * (len(rows) + 2))
        fig.text(0.07, 0.905, "Radien", fontsize=11.5, weight="bold")
        ax = fig.add_axes([0.05, 0.895 - h1, 0.9, h1])
        _table(ax, header, rows, col_widths=[0.19, 0.08] + [0.21] * n + [0.13],
               fontsize=8.5, first_col_left=True)

        # Tabelle 2: Uniformity und Crosstalk (hart je Region + atomgewichtet)
        y2 = 0.895 - h1 - 0.045
        fig.text(0.07, y2 + 0.01, "Uniformity und Crosstalk (%)", fontsize=11.5, weight="bold")
        header = ["Datei"] + [f"U {LEVEL_NAMES.get(l, l)}" for l in labels] + ["U atom"]
        header += [f"$\\eta$ {LEVEL_NAMES.get(l, l)}" for l in labels] + ["$\\eta$ atom"]
        rows = []
        for r in results:
            s = r.summary()
            L = r.lattice
            row = [r.name] + [_c(s[f"U_{l}_pct"]) for l in labels]
            row += [_c(100 * L["U_w"]) if L else "–"]
            row += [_c(100 * L["hard"][l]["eta"]) if L else "–" for l in labels]
            row += [_c(100 * L["eta_w"]) if L else "–"]
            rows.append(row)
        h2 = min(0.3, 0.021 * (len(rows) + 2))
        ax = fig.add_axes([0.05, y2 - h2, 0.9, h2])
        _table(ax, header, rows, col_widths=[0.2] + [0.1] * (2 * n + 2), fontsize=8.5,
               first_col_left=True)
        h = (0.9 - (y2 - h2))

        # Balkendiagramm: Uniformity je Region und Bild
        top = 0.9 - h - 0.08
        axb = fig.add_axes([0.12, max(0.12, top - 0.3), 0.8, 0.28])
        x = np.arange(len(results))
        bw = 0.8 / n
        for i, l in enumerate(labels):
            vals = [r.summary()[f"U_{l}_pct"] for r in results]
            axb.bar(x + (i - (n - 1) / 2) * bw, vals, bw * 0.92,
                    color=pl._level_color(l, i), edgecolor="#333333", linewidth=0.8,
                    label=LEVEL_NAMES.get(l, l))
        axb.set_xticks(x)
        axb.set_xticklabels([r.name for r in results], rotation=30 if len(results) > 4 else 0,
                            ha="right" if len(results) > 4 else "center")
        axb.set_ylabel(r"Uniformity $\sigma/\mu$ (%)")
        axb.set_title("Uniformity je Region", pad=26)
        axb.legend(ncol=n, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
        allv = [r.summary()[f"U_{l}_pct"] for r in results for l in labels]
        if np.isfinite(np.nanmax(allv)):
            axb.set_ylim(0, 1.08 * np.nanmax(allv))
        fig.text(0.07, 0.035,
                 "w: Radius bis zum Level (Durchmesser = 2w), lang/kurz entlang der "
                 "Ellipsen-Halbachsen. r mit U < " + thr + " %: größter Kreis um das Zentrum, "
                 "in dem $\\sigma/\\mu$ unter der Schwelle bleibt.\n"
                 "U: $\\sigma/\\mu$ innerhalb der Level-Ellipse,\n"
                 "$\\eta$: Crosstalk der 8 Nachbarn (Pitch siehe Einzelseiten); "
                 "atom: mit thermischer Atomverteilung gewichtet. '–' = nicht berechnet.\n"
                 "Details je Bild auf den folgenden Seiten.",
                 fontsize=7.2, color=MUTED, va="bottom", linespacing=1.45)
    return fig


def compact(res: FlatTopResult, max_px: int = 800) -> FlatTopResult:
    """Speichersparende Kopie fuer den Report (Batch mit vielen grossen Bildern):
    2D-Ausschnitt auf <= max_px Pixel gemittelt, Fit-ROI verworfen."""
    import copy
    r = copy.copy(res)
    v = res.view_data
    b = int(math.ceil(max(v.shape) / max_px))
    if b > 1:
        ny, nx = (v.shape[0] // b) * b, (v.shape[1] // b) * b
        y0, y1, x0, x1 = res.view
        v = v[:ny, :nx].reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))
        r.view = (y0, y0 + ny, x0, x0 + nx)
    r.view_data = v.astype(np.float32)
    r.roi_data = r.roi_model = None
    return r


def save_report(results: list[FlatTopResult], path: str | Path,
                plot_opts: dict | None = None) -> Path:
    """Schreibt den Report. Bei mehreren Bildern mit Vergleichsseite vorne."""
    path = Path(path)
    plot_opts = plot_opts or {}
    results = [r for r in results if r is not None]
    with PdfPages(path) as pdf:
        with pl._rc():
            if len(results) > 1:
                pdf.savefig(_summary_page(results, str(Path(results[0].path).parent)))
            for res in results:
                pdf.savefig(_page_for(res, plot_opts))
        d = pdf.infodict()
        d["Title"] = "FlatTop-Auswertung"
        d["Creator"] = "BeamShaperPlots"
    return path
