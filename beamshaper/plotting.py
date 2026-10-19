"""Abbildungen aus einem FlatTopResult. Hier wird NICHT gerechnet.

Plots haben KEINEN Titel (Vorgabe fuer die Auswertung).
Alle Funktionen zeichnen in eine uebergebene matplotlib-Figure, damit dieselbe
Funktion fuer die GUI (FigureCanvas) und fuer den PDF-Export benutzt wird.

Stil nach Vorlage AOD_Ellipse.pdf:
  - links das 2D-Bild ohne Achsen, jet, schwarzes Fadenkreuz durchs Zentrum,
    optional die drei Regionen (1/e^2, 50 %, 90 %) als gestrichelte Ellipsen
    mit Legende oben und optional weisser Text unten rechts (FWHM lang/kurz)
  - rechts horizontaler und vertikaler Schnitt (Rohintensitaet), grauer
    Hintergrund mit weissem Gitter, Levels als gestrichelte Linien
  - Serifenschrift, Einheiten immer in runden Klammern
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Circle

from .analysis import FlatTopResult, level_labels

# Regionen-Farben wie in der Vorlage: 1/e^2, 50 %, 90 %
# Regionen-Farben (Vorgabe): Pink aussen (1/e2), Schwarz mittig (50 %),
# Cyan innen (90 %). Dazu ein duenner Kontrastrand, damit jede Linie im 2D-Bild
# und auf dem grauen Hintergrund der Schnitte sichtbar bleibt.
LEVEL_COLORS = {"1/e2": "#FF33CC", "50%": "#000000", "90%": "#00FFFF"}
LEVEL_EDGES = {"1/e2": "#000000", "50%": "#FFFFFF", "90%": "#000000"}
EXTRA_COLORS = ["#FFD000", "#8C8C8C", "#00E5FF"]
DATA_COLOR = "blue"
FIT_COLOR = "black"
PANEL_BG = "#EAEAF2"      # seaborn-darkgrid

# Schriftgroesse in pt - EINE Groesse fuer alle Texte (Achsen, Ticks, Legenden,
# Titel der Teilplots, Text im Bild). Die Figuren werden in 16 cm Breite
# (\textwidth) exportiert; mit \includegraphics[width=\textwidth]{...} erscheint
# die Schrift in LaTeX also exakt in dieser Groesse (Standard 11 pt = Fliesstext).
FONT_PT = 11.0
TEXTWIDTH_IN = 16 / 2.54


def set_font_size(pt: float) -> None:
    global FONT_PT
    FONT_PT = float(pt)


import matplotlib as _mpl

MPL_VERSION = tuple(int(x) for x in _mpl.__version__.split(".")[:2])
MPL_RECOMMENDED = (3, 7)


def _layout(fig: Figure, constrained: bool = True, **pads) -> None:
    """Layout-Engine setzen - auch fuer aeltere matplotlib-Versionen (< 3.6)."""
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine("constrained" if constrained else "none", **pads)
    else:
        fig.set_constrained_layout(constrained)
        if constrained and pads:
            fig.set_constrained_layout_pads(**pads)


FONT_DIR = Path(__file__).resolve().parent / "fonts"


def _latex_like_fonts() -> list[str]:
    """LaTeX-Schrift Latin Modern Roman (liegt dem Paket bei, GUST Font License).

    WICHTIG: NICHT cmr10 verwenden - dort liegt auf dem Code von 'µ' ein
    Gravis-Akzent, im PDF stand dann "('m)" statt "(µm)".
    """
    from matplotlib import font_manager
    for f in sorted(FONT_DIR.glob("*.otf")):
        try:
            font_manager.fontManager.addfont(str(f))
        except Exception:
            pass
    have = {f.name for f in font_manager.fontManager.ttflist}
    fams = [f for f in ("Latin Modern Roman", "DejaVu Serif") if f in have]
    return fams or ["serif"]


STYLE = {
    "font.family": _latex_like_fonts(),
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": True,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": "white",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "white",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 3,
}

LEVEL_TEX = {"1/e2": r"$1/e^2$", "50%": "50%", "90%": "90%"}
INT_LABEL = "Intensity (a. u.)"
INT_LABEL_NORM = "Intensity (norm.)"
INT_LABEL_REL = r"$I/I_\mathrm{max}$"      # wie in den Simulationsplots
POS_LABEL = "Position (µm)"


def _tex(label: str) -> str:
    return LEVEL_TEX.get(label, label)


def _level_color(label: str, i: int = 0) -> str:
    return LEVEL_COLORS.get(label, EXTRA_COLORS[i % len(EXTRA_COLORS)])


def _level_fx(label: str, lw: float):
    """Kontrastrand um Ringe/Linien, damit sie auf jedem Untergrund sichtbar sind."""
    import matplotlib.patheffects as pe
    edge = LEVEL_EDGES.get(label, "#000000")
    return [pe.Stroke(linewidth=lw + 0.8, foreground=edge), pe.Normal()]


def _comma(v: float, dec: int = 2) -> str:
    return f"{v:.{dec}f}".replace(".", ",")


def _extent_um(res: FlatTopResult, box=None):
    """imshow-extent (ROI oder box) in um relativ zum Fitzentrum (y nach unten wie im Bild)."""
    y0, y1, x0, x1 = box if box is not None else res.roi
    s, f = res.um_per_px, res.fit
    return [(x0 - 0.5 - f.x0) * s, (x1 - 0.5 - f.x0) * s,
            (y1 - 0.5 - f.y0) * s, (y0 - 0.5 - f.y0) * s]


def _cut_halfwidth(res: FlatTopResult) -> float:
    return float(np.max(np.abs(next(iter(res.cuts.values())).pos_um)))


# Intensitaetsskala: "bitdepth" = 0 ... max. Pixelwert der Bittiefe (Standard),
#                     "image"    = 0 ... hellster Pixel des jeweiligen Bildes
SCALE_MODE = "bitdepth"
# y-Achse der Schnitte: "follow" = wie die Farbskala, sonst eigene Wahl.
# So kann das 2D-Bild auf das hellste Pixel gespreizt werden (dunkle Aufnahmen
# wie FlatTop_05 sind dann ueberhaupt erkennbar), waehrend die Schnitte bis zum
# maximalen Pixelwert laufen und damit zwischen Bildern vergleichbar bleiben.
CUT_SCALE_MODE = "follow"


def set_scale_mode(mode: str, cuts: str | None = None) -> None:
    global SCALE_MODE, CUT_SCALE_MODE
    SCALE_MODE = mode if mode in ("bitdepth", "image") else "bitdepth"
    if cuts is not None:
        CUT_SCALE_MODE = (cuts if cuts in ("bitdepth", "image", "follow", "norm")
                          else "follow")


def _int_max(res: FlatTopResult, cuts: bool = False) -> float:
    """Obergrenze aller Intensitaetsskalen.

    Standard: maximaler Pixelwert der Bittiefe (255 bei 8 bit, 4095 bei 12 bit,
    65535 bei 16 bit) - fest, damit Aufnahmen gleicher Bittiefe (z. B.
    FlatTop_04 und FlatTop_05) direkt vergleichbar sind.
    """
    mode = SCALE_MODE
    if cuts and CUT_SCALE_MODE != "follow":
        mode = CUT_SCALE_MODE
    if mode == "image":
        return float(max(res.max_value, 1.0))
    return float(res.saturation_value)


def _cut_values(res: FlatTopResult, norm_data):
    """Rohintensitaet (Counts) oder auf 1 normiert.

    normiert: (I - Untergrund) / (Fit-Peak - Untergrund). Untergrund ist der
    Bildrand-Mittelwert, nicht der Fit-Offset B - so liegt die Basislinie auf 0
    und das Plateau auf 1.
    """
    f = res.fit
    raw = norm_data * f.A + f.B
    if CUT_SCALE_MODE != "norm":
        return raw
    span = f.A + f.B - res.background
    return (raw - res.background) / (span if span > 0 else 1.0)


def _draw_cut(ax, res, cut, labels, show_regions=True, show_fit=False):
    """Schnitt in Rohintensitaet (counts) wie in der Vorlage, oder normiert."""
    ax.plot(cut.pos_um, _cut_values(res, cut.data), color=DATA_COLOR, lw=1.3, label="Data",
            zorder=4)
    if show_fit:
        ax.plot(cut.pos_um, _cut_values(res, cut.model), color=FIT_COLOR, lw=1.0, ls="--",
                label="Fit")
    if show_regions:
        for i, lab in enumerate(labels):
            r = cut.level_radii_um[lab]
            for sgn in (-1, 1):
                ax.axvline(sgn * r, color=_level_color(lab, i), lw=1.5, ls=(0, (4, 2)),
                           path_effects=_level_fx(lab, 1.5), zorder=2, gid="region")
    ax.set_title(cut.name)
    if CUT_SCALE_MODE == "norm":
        ax.set_ylabel(INT_LABEL_NORM)
        ax.set_ylim(0, 1.1)
    else:
        ax.set_ylabel(INT_LABEL)
        ax.set_ylim(0, _int_max(res, cuts=True))


def _region_handles(labels):
    """Legenden-Eintraege als gestrichelte Kaestchen wie in der Vorlage."""
    from matplotlib.patches import Rectangle
    h = [Rectangle((0, 0), 1, 1, fill=False, lw=2.0, ls="--",
                   ec=_level_color(l, i), label=_tex(l),
                   path_effects=_level_fx(l, 2.0)) for i, l in enumerate(labels)]
    # matplotlib fuellt Legenden spaltenweise -> umsortieren, damit zeilenweise
    # (1/e^2, 50 % | 90 %) wie in der Vorlage
    ncol = 2
    nrow = -(-len(h) // ncol)
    rows = [h[i:i + ncol] for i in range(0, len(h), ncol)]
    return [rows[r][c] for c in range(ncol) for r in range(nrow) if c < len(rows[r])]


# ---------------------------------------------------------------------------
def plot_overview(fig: Figure, res: FlatTopResult, cmap: str = "jet",
                  show_regions: bool = True, show_text: bool = True,
                  show_fit: bool = False, show_circles: bool = False,
                  show_scalebar: bool = True) -> None:
    """2D-Bild links, horizontaler und vertikaler Schnitt rechts."""
    fig.clear()
    _layout(fig, w_pad=0.04, h_pad=0.04, wspace=0.04)
    with _rc():
        gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1])
        ax_im = fig.add_subplot(gs[:, 0])
        ax_h = fig.add_subplot(gs[0, 1])
        ax_v = fig.add_subplot(gs[1, 1], sharex=ax_h, sharey=ax_h)
        draw_overview(ax_im, ax_h, ax_v, res, cmap, show_regions, show_text,
                      show_fit, show_circles, show_scalebar)


def _scalebar(ax, xl, xr, yb, yt):
    """Weisser Massstabsbalken unten links (das 2D-Bild hat keine Achsen)."""
    width = xr - xl
    target = width / 5
    exp = 10 ** math.floor(math.log10(target))
    length = max(m * exp for m in (1, 2, 5) if m * exp <= target)
    x0 = xl + 0.05 * width
    y0 = yb - 0.06 * (yb - yt)
    ax.plot([x0, x0 + length], [y0, y0], color="white", lw=3, solid_capstyle="butt",
            clip_on=True)
    lab = f"{length:g} µm".replace(".", ",")
    ax.text(x0 + length / 2, y0 - 0.02 * (yb - yt), lab, color="white", ha="center",
            va="bottom", clip_on=True)


def draw_overview(ax_im, ax_h, ax_v, res: FlatTopResult, cmap: str = "jet",
                  show_regions: bool = True, show_text: bool = True,
                  show_fit: bool = False, show_circles: bool = False,
                  show_scalebar: bool = True) -> None:
    """Zeichnet die Uebersicht in drei vorhandene Achsen (auch fuer den Report)."""
    labels = level_labels(res.settings)
    f, s = res.fit, res.um_per_px
    hw = _cut_halfwidth(res)

    # --- 2D-Bild ---------------------------------------------------------
    ax_im.imshow(res.view_data, cmap=cmap, extent=_extent_um(res, res.view),
                 vmin=0, vmax=_int_max(res),
                 interpolation="nearest", aspect="equal")
    # Ausschnitt = Schnitt aus gewuenschtem Bereich und tatsaechlichem Kamerabild,
    # damit das Bild die Achse immer vollstaendig fuellt (kein weisser Rand)
    ex0, ex1, ey1, ey0 = _extent_um(res, res.view)
    xl, xr = max(-hw, ex0), min(hw, ex1)
    yt, yb = max(-hw, ey0), min(hw, ey1)
    ax_im.set_xlim(xl, xr)
    ax_im.set_ylim(yb, yt)
    ax_im.set_axis_off()               # wie Vorlage: 2D-Bild ohne Achsen
    # Fadenkreuz nur innerhalb des Bildes
    ax_im.plot([xl, xr], [0, 0], color="black", lw=0.6, ls="--", clip_on=True)
    ax_im.plot([0, 0], [yt, yb], color="black", lw=0.6, ls="--", clip_on=True)

    if show_regions:
        for i, lab in enumerate(labels):
            a, b = res.radii_px[lab]
            ax_im.add_patch(Ellipse((0, 0), 2 * a * s, 2 * b * s,
                                    angle=math.degrees(f.theta), fill=False, lw=2.4,
                                    ls="--", ec=_level_color(lab, i),
                                    path_effects=_level_fx(lab, 2.4), gid="region"))
        # Legende vollstaendig im 2D-Bild
        leg = ax_im.legend(handles=_region_handles(labels), loc="upper center",
                           ncol=2, frameon=True, handlelength=1.8, handleheight=0.9,
                           columnspacing=1.2, borderpad=0.4, borderaxespad=0.6)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("#55555F")
        leg.get_frame().set_alpha(1.0)

    if show_circles and res.flatness_d_um.size:
        for d in np.linspace(res.flatness_d_um[0], res.flatness_d_um[-1], 5)[1:]:
            ax_im.add_patch(Circle((0, 0), d / 2, fill=False, lw=1.0, ec="white", alpha=0.8))

    if show_scalebar:
        _scalebar(ax_im, xl, xr, yb, yt)

    if show_text:
        sm = res.summary()
        txt = (f"FWHM (width): {_comma(sm['D_long_50%_um'])} µm\n"
               f"FWHM (height): {_comma(sm['D_short_50%_um'])} µm")
        ax_im.text(0.97, 0.03, txt, transform=ax_im.transAxes, ha="right", va="bottom",
                   color="white", linespacing=1.3, clip_on=True)

    # --- Schnitte --------------------------------------------------------
    _draw_cut(ax_h, res, res.cuts["horizontal"], labels, show_regions, show_fit)
    _draw_cut(ax_v, res, res.cuts["vertikal"], labels, show_regions, show_fit)
    ax_h.set_title("Horizontal")
    ax_v.set_title("Vertical")
    ax_h.set_xlim(-hw, hw)
    ax_v.set_xlabel(POS_LABEL)
    ax_h.tick_params(labelbottom=True)
    if show_fit:
        ax_h.legend(loc="upper right", frameon=True)


def plot_axis_cuts(fig: Figure, res: FlatTopResult, show_regions: bool = True,
                   show_fit: bool = True) -> None:
    """Schnitte entlang langer und kurzer Achse (geteilte Achsen)."""
    fig.clear()
    _layout(fig)
    with _rc():
        ax1, ax2 = fig.subplots(1, 2, sharex=True, sharey=True)
        labels = level_labels(res.settings)
        hw = _cut_halfwidth(res)
        _draw_cut(ax1, res, res.cuts["lange Achse"], labels, show_regions, show_fit)
        _draw_cut(ax2, res, res.cuts["kurze Achse"], labels, show_regions, show_fit)
        ax1.set_title("Long axis")
        ax2.set_title("Short axis")
        ax2.set_ylabel("")
        ax1.set_xlim(-hw, hw)
        for ax in (ax1, ax2):
            ax.set_xlabel(POS_LABEL)
        handles = [Line2D([], [], color=DATA_COLOR, lw=1.2, label="Data")]
        if show_fit:
            handles.append(Line2D([], [], color=FIT_COLOR, lw=1.0, ls="--", label="Fit"))
        if show_regions:
            handles += [Line2D([], [], color=_level_color(l, i), ls="--", lw=1.6, label=_tex(l),
                               path_effects=_level_fx(l, 1.6))
                        for i, l in enumerate(labels)]
        leg_kw = dict(handles=handles, ncol=len(handles), frameon=False,
                      handlelength=1.6, columnspacing=1.2)
        if MPL_VERSION >= (3, 7):
            fig.legend(loc="outside lower center", **leg_kw)
        else:   # aeltere matplotlib: festes Layout, Legende unten mittig
            _layout(fig, constrained=False)
            fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.3, wspace=0.08)
            fig.legend(loc="lower center", **leg_kw)


def plot_fit_residual(fig: Figure, res: FlatTopResult, cmap: str = "jet") -> None:
    """Daten | Fit | Residuum (in % von A)."""
    fig.clear()
    _layout(fig)
    with _rc():
        axs = fig.subplots(1, 3, sharex=True, sharey=True)
        f = res.fit
        ext = _extent_um(res)
        resid = 100 * (res.roi_data - res.roi_model) / f.A
        lim = float(np.nanpercentile(np.abs(resid), 99.5)) or 1.0
        vmax = _int_max(res)
        for ax, arr, title, cm, vmin, vm, lab in (
            (axs[0], res.roi_data, "Data", cmap, 0, vmax, INT_LABEL),
            (axs[1], res.roi_model, "Supergaussian fit", cmap, 0, vmax, INT_LABEL),
            (axs[2], resid, "Residual", "RdBu_r", -lim, lim, r"Residual (% of $A$)"),
        ):
            im = ax.imshow(arr, cmap=cm, extent=ext, vmin=vmin, vmax=vm, interpolation="nearest")
            ax.grid(False)
            ax.set_title(title)
            ax.set_xlabel(r"$x$ (µm)")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, orientation="horizontal")
            cb.set_label(lab)
            cb.outline.set_visible(False)
        axs[0].set_ylabel(r"$y$ (µm)")


def plot_flatness(fig: Figure, res: FlatTopResult, show_regions: bool = True) -> None:
    """Uniformity std/mean in Kreisen wachsenden Durchmessers."""
    fig.clear()
    _layout(fig)
    with _rc():
        ax = fig.add_subplot(111)
        ax.plot(res.flatness_d_um, 100 * res.flatness_u, color=DATA_COLOR, lw=1.4,
                marker="o", ms=3)
        if show_regions:
            for i, lab in enumerate(level_labels(res.settings)):
                a, b = res.radii_px[lab]
                ax.axvline(2 * b * res.um_per_px, color=_level_color(lab, i), ls="--", lw=1.6,
                           path_effects=_level_fx(lab, 1.6), gid="region",
                           label=f"{_tex(lab)}, short axis")
            ax.legend(loc="upper left", frameon=True, facecolor="white")
        # Schwelle (Standard 5 %) und der zugehoerige Radius
        thr = 100 * res.settings.uniformity_threshold
        r_u = res.uniformity_radius_um
        ax.axhline(thr, color="#55555F", ls=":", lw=1.2)
        ax.text(0.01, thr, f" {_comma(thr, 0)} %", transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", color="#55555F")
        if np.isfinite(r_u):
            ax.plot([2 * r_u], [thr], marker="o", ms=6, mfc="none", mew=1.6,
                    color=DATA_COLOR)
            ax.annotate(f"r = {_comma(r_u)} µm", xy=(2 * r_u, thr),
                        xytext=(8, 10), textcoords="offset points", color=DATA_COLOR)
        ax.set_xlabel("Circle diameter (µm)")
        ax.set_ylabel(r"Uniformity $\sigma/\mu$ (%)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)


# ---------------------------------------------------------------------------
# Vergleich zweier Profile (z. B. Simulation aus Multitone_Simulation und Messung)
# ---------------------------------------------------------------------------
def comparison_order(res_a: FlatTopResult, res_b: FlatTopResult):
    """Reihenfolge und Beschriftung fuer den Vergleich: Simulation links.

    Rueckgabe: [(res, label), (res, label)]."""
    pair = [res_a, res_b]
    if res_b.is_simulation and not res_a.is_simulation:
        pair = [res_b, res_a]

    def label(r):
        if r.is_simulation and not (res_a.is_simulation and res_b.is_simulation):
            return "Simulation"
        if not r.is_simulation and (res_a.is_simulation or res_b.is_simulation):
            return "Measurement"
        return r.name
    return [(r, label(r)) for r in pair]


def _norm_view(res: FlatTopResult):
    """Darstellungsausschnitt auf 0 ... 1 normiert: (I - Untergrund) / (Peak - Untergrund).

    Peak = Maximum nach leichter Glaettung (1 px), damit ein einzelnes
    Rauschpixel im Kamerabild die Skala nicht bestimmt. Untergrund = Bildrand-
    Mittel wie in der Auswertung (Simulation: praktisch 0)."""
    from scipy import ndimage
    v = np.asarray(res.view_data, dtype=float)
    bg = float(res.background)
    peak = float(ndimage.gaussian_filter(v, 1.0).max())
    span = peak - bg if peak > bg else 1.0
    return (v - bg) / span


def comparison_halfwidth(res_a: FlatTopResult, res_b: FlatTopResult) -> float:
    """Gemeinsame Halbbreite (um) des Ausschnitts: so gross wie der groessere der
    beiden Plotausschnitte (Nebenstruktur sichtbar), aber nicht groesser als das,
    was beide Bilder um ihr Zentrum hergeben (kein leerer Rand)."""
    want = max(_cut_halfwidth(res_a), _cut_halfwidth(res_b))
    have = []
    for r in (res_a, res_b):
        x0, x1, y1, y0 = _extent_um(r, r.view)
        have.append(min(-x0, x1, -y0, y1))
    return float(min(want, *have))


def plot_comparison(fig: Figure, res_a: FlatTopResult, res_b: FlatTopResult,
                    cmap: str = "jet", half_um: float | None = None,
                    show_crosshair: bool = False) -> None:
    """Zwei 2D-Profile nebeneinander, gleicher Ausschnitt (um) um das jeweilige
    Zentrum, jedes auf sein Maximum normiert (I/I_max), gemeinsame Farbskala
    so hoch wie die Bilder. Ohne Ueberschriften.

    Simulation steht links. Achsen in um relativ zum Fitzentrum, y nach oben.
    """
    fig.clear()
    _layout(fig, w_pad=0.04, h_pad=0.04, wspace=0.06)
    order = comparison_order(res_a, res_b)
    hw = half_um if half_um else comparison_halfwidth(res_a, res_b)
    with _rc():
        axs = fig.subplots(1, 2, sharex=True, sharey=True)
        im = None
        for ax, (r, lab) in zip(axs, order):
            x0, x1, yb, yt = _extent_um(r, r.view)        # y nach unten (Bild)
            im = ax.imshow(_norm_view(r), cmap=cmap, vmin=0, vmax=1,
                           extent=[x0, x1, -yb, -yt],   # y nach oben
                           interpolation="nearest", aspect="equal")
            ax.grid(False)
            ax.set_xlim(-hw, hw)
            ax.set_ylim(-hw, hw)
            ax.set_xlabel(r"$x$ (µm)")
            if show_crosshair:
                ax.axhline(0, color="black", lw=0.6, ls="--")
                ax.axvline(0, color="black", lw=0.6, ls="--")
        axs[0].set_ylabel(r"$y$ (µm)")
        from matplotlib.ticker import MaxNLocator
        for loc in (axs[0].xaxis, axs[0].yaxis):
            loc.set_major_locator(MaxNLocator(nbins=5, integer=True, symmetric=True))
        # Farbskala exakt so hoch wie die Bildflaeche: als Inset-Achse in
        # Achsenkoordinaten des rechten Bildes - folgt dessen tatsaechlicher
        # Position (aspect="equal"), auch beim Vergroessern des GUI-Fensters.
        cax = axs[1].inset_axes([1.05, 0.0, 0.05, 1.0])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(INT_LABEL_REL)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0)


# ---------------------------------------------------------------------------
# Titelbild fuer die Titelseite der Thesis (APQ-/TUDa-Vorlage)
# ---------------------------------------------------------------------------
# Bildfenster der Titelseite: volle Textbreite (A4, inner 31 mm, outer 24 mm)
# und vom Strich unter dem Titelblock bis zum Strich unten. Gemessen in
# APQ_Thesis_Vorlage.pdf: 155,0 mm x 184,9 mm (Strich-Unterkante bis
# Strich-Unterkante, die Striche liegen ueber dem Bild).
TITLE_W_MM = 155.0
TITLE_H_MM = 184.9
# Logo-Kaesten oben rechts im Fenster (mm, von links oben): x0, y0, x1, y1.
# Nur als Hilfe in der GUI-Vorschau, wird nicht exportiert.
TITLE_LOGO_BOX_MM = (106.6, 11.0, TITLE_W_MM, 54.5)

_RAW_CACHE: dict = {}


def _raw_image(res: FlatTopResult, subtract_bg: bool = False):
    """Kamerabild fuer das Titelbild und Offset (oy, ox).

    Standard: unbearbeitet (Bilddatei selbst, gecacht nach Pfad + Aenderungszeit).
    subtract_bg=True: das in den Einstellungen gewaehlte Hintergrundbild wird
    abgezogen (negative Werte -> 0); ohne Hintergrundbild bleibt es unbearbeitet.
    Gibt es keine Datei (z. B. synthetisches Bild in den Tests), wird der
    gespeicherte Darstellungsausschnitt verwendet.
    """
    from .imageio import load_image
    p = Path(res.path) if res.path else None
    if p is None or not p.is_file():
        return res.view_data, (res.view[0], res.view[2])
    bgp = res.settings.background_path if subtract_bg else None
    bgp = Path(bgp) if bgp and Path(bgp).is_file() else None
    key = (str(p), p.stat().st_mtime_ns,
           (str(bgp), bgp.stat().st_mtime_ns) if bgp else None)
    if key not in _RAW_CACHE:
        img = load_image(p)[0]
        if bgp is not None:
            bg = load_image(bgp)[0]
            if bg.shape != img.shape:
                raise ValueError(f"Hintergrundbild {bg.shape} passt nicht zu {img.shape}")
            img = np.clip(img - bg, 0, None)
        _RAW_CACHE.clear()
        _RAW_CACHE[key] = img
    return _RAW_CACHE[key], (0, 0)


def title_object_box(img: np.ndarray, offset=(0, 0), level: float = 0.135):
    """Umriss des Strahls bzw. Musters (z. B. 3x3-Spotgitter) im Bild.

    Leicht geglaettet; alle Pixel ueber `level` (Standard 1/e^2) zwischen Untergrund
    (Median) und Maximum. Kleine Rauschinseln (< 5 % der groessten Flaeche) zaehlen
    nicht. Rueckgabe: (cx, cy, Breite, Hoehe) in Pixeln (Bildkoordinaten) oder None.
    """
    from scipy import ndimage
    b = max(1, int(max(img.shape) // 1000))          # grosse Bilder fuer die Suche binnen
    small = img[: img.shape[0] // b * b, : img.shape[1] // b * b]
    small = small.reshape(small.shape[0] // b, b, small.shape[1] // b, b).mean((1, 3))
    sm = ndimage.gaussian_filter(small, 2)
    base, peak = float(np.median(sm)), float(sm.max())
    if peak <= base:
        return None
    lab, n = ndimage.label(sm > base + level * (peak - base))
    if n == 0:
        return None
    area = np.bincount(lab.ravel())[1:]
    keep = np.isin(lab, 1 + np.flatnonzero(area >= 0.05 * area.max()))
    ys, xs = np.nonzero(keep)
    oy, ox = offset
    x0, x1 = xs.min() * b, (xs.max() + 1) * b
    y0, y1 = ys.min() * b, (ys.max() + 1) * b
    return (ox + (x0 + x1) / 2 - 0.5, oy + (y0 + y1) / 2 - 0.5, x1 - x0, y1 - y0)


def title_window(res: FlatTopResult, img: np.ndarray, fill: float = 0.45, pos_x: float = 0.5,
                 pos_y: float = 0.5, w_mm: float = TITLE_W_MM, h_mm: float = TITLE_H_MM,
                 offset=(0, 0)):
    """Ausschnitt (links, oben, Breite, Hoehe) in Pixelkoordinaten fuer das Titelbild.

    - Seitenverhaeltnis exakt h_mm / w_mm, quadratische Pixel
    - der Strahl bzw. das ganze Muster (Umriss bei 1/e^2, s. title_object_box)
      nimmt `fill` der Fensterbreite ein (bzw. der Hoehe, falls es hoch ist)
    - dessen Mitte liegt bei (pos_x, pos_y) im Fenster (0,5/0,5 = Mitte)
    - das Fenster liegt immer vollstaendig im Kamerabild (kein Rand); ist das Bild
      zu klein, wird das Fenster verkleinert (Strahl dann groesser als gewuenscht)
    Rueckgabe: (x0, y0, W, H, verkleinert) - Pixelkanten, Pixel i reicht von i-0,5 bis i+0,5.
    """
    ny, nx = img.shape[:2]
    oy, ox = offset
    ratio = h_mm / w_mm
    box = title_object_box(img, offset)
    if box is None:                      # Rueckfall: Fit
        a = max(a for a, _ in res.radii_px.values())
        box = (res.fit.x0, res.fit.y0, 2 * a, 2 * a)
    cx, cy, bw, bh = box
    W = max(bw, bh / ratio) / max(fill, 1e-3)
    H = W * ratio
    shrunk = False
    s = min(nx / W, ny / H, 1.0)
    if s < 1.0:
        W, H, shrunk = W * s, H * s, True
    x0 = cx - pos_x * W
    y0 = cy - pos_y * H
    x0 = float(np.clip(x0, ox - 0.5, ox + nx - 0.5 - W))
    y0 = float(np.clip(y0, oy - 0.5, oy + ny - 0.5 - H))
    return x0, y0, W, H, shrunk


def plot_title(fig: Figure, res: FlatTopResult, cmap: str = "jet",
               title_fill: float = 0.45, title_x: float = 0.5, title_y: float = 0.5,
               title_w_mm: float = TITLE_W_MM, title_h_mm: float = TITLE_H_MM,
               title_bg: bool = False, show_logo_hint: bool = True) -> None:
    """Unbearbeitetes Kamerabild im Farbschema, randlos, fuer die Titelseite.

    Kein Fadenkreuz, keine Ringe, keine Achsen, kein Text. Hintergrundabzug nur mit
    title_bg=True (Hintergrundbild aus den Einstellungen).
    Export: Figur exakt title_w_mm x title_h_mm, das Bild fuellt sie vollstaendig.
    In LaTeX:  \\titleimage*{\\includegraphics{<name>_title.pdf}}
    (die Stern-Variante skaliert/beschneidet auf das Fenster zwischen den Strichen).
    """
    fig.clear()
    _layout(fig, constrained=False)
    img, (oy, ox) = _raw_image(res, title_bg)
    x0, y0, W, H, shrunk = title_window(res, img, title_fill, title_x, title_y,
                                        title_w_mm, title_h_mm, offset=(oy, ox))
    # nur den benoetigten Teil (+1 px) einbetten -> kleines PDF (Indizes im Array)
    ny, nx = img.shape
    ix0 = max(0, int(np.floor(x0 - ox + 0.5)) - 1)
    ix1 = min(nx, int(np.ceil(x0 - ox + W + 0.5)) + 1)
    iy0 = max(0, int(np.floor(y0 - oy + 0.5)) - 1)
    iy1 = min(ny, int(np.ceil(y0 - oy + H + 0.5)) + 1)
    cm = _mpl.colormaps[cmap] if hasattr(_mpl, "colormaps") else _mpl.cm.get_cmap(cmap)

    # Export: Hintergrund in der untersten Farbe der Skala - selbst ein Rundungs-
    # Haarstrich am Rand waere dann nicht weiss. Vorschau: grau, damit man das
    # Fenster sieht.
    fig.patch.set_facecolor("#B4B4B4" if show_logo_hint else cm(0.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img[iy0:iy1, ix0:ix1], cmap=cm, vmin=0, vmax=_int_max(res),
              extent=(ox + ix0 - 0.5, ox + ix1 - 0.5, oy + iy1 - 0.5, oy + iy0 - 0.5),
              interpolation="none", aspect="equal")
    ax.set_xlim(x0, x0 + W)
    ax.set_ylim(y0 + H, y0)
    ax.set_axis_off()

    if show_logo_hint:
        from matplotlib.patches import Rectangle
        k = W / title_w_mm                       # px pro mm
        lx0, ly0, lx1, ly1 = TITLE_LOGO_BOX_MM
        lx0, lx1 = lx0 * title_w_mm / TITLE_W_MM, lx1 * title_w_mm / TITLE_W_MM
        ax.add_patch(Rectangle((x0 + lx0 * k, y0 + ly0 * k), (lx1 - lx0) * k,
                               (ly1 - ly0) * k, fc="white", ec="black", ls="--",
                               alpha=0.75, lw=1.0))
        ax.text(x0 + (lx0 + lx1) / 2 * k, y0 + (ly0 + ly1) / 2 * k,
                "Logos\n(nur Vorschau)", ha="center", va="center", fontsize=9)
        note = f"{_comma(title_w_mm, 1)} × {_comma(title_h_mm, 1)} mm"
        if shrunk:
            note += " – Kamerabild zu klein, Ausschnitt verkleinert"
        ax.text(0.01, 0.01, note, transform=ax.transAxes, color="white", fontsize=9,
                ha="left", va="bottom")


def title_figsize(w_mm: float = TITLE_W_MM, h_mm: float = TITLE_H_MM) -> tuple:
    return (w_mm / 25.4, h_mm / 25.4)


FIGURES = {
    "overview": ("Übersicht", plot_overview),
    "axes": ("Achsenschnitte", plot_axis_cuts),
    "fit": ("Fit & Residuum", plot_fit_residual),
    "flatness": ("Flatness", plot_flatness),
    "title": ("Titelbild", plot_title),
}
# Standard-Export (Einzelbild + Batch) ohne das Titelbild; das wird gezielt gespeichert
DEFAULT_EXPORT = ["overview", "axes", "fit", "flatness"]

# Exportgroessen in Zoll; overview im Seitenverhaeltnis der Vorlage (1382 x 731 pt)
EXPORT_SIZES = {          # alle 16 cm breit (\textwidth), Hoehe in Zoll
    "overview": (TEXTWIDTH_IN, TEXTWIDTH_IN * 730.8 / 1382.4),   # Format der Vorlage
    "axes": (TEXTWIDTH_IN, 2.9),
    "fit": (TEXTWIDTH_IN, 3.5),
    "flatness": (TEXTWIDTH_IN, 3.2),
    "title": title_figsize(),        # wird beim Export aus title_w_mm/title_h_mm gesetzt
    "comparison": (TEXTWIDTH_IN, 3.05),
}


class _rc:
    """Kontextmanager fuer den Plotstil, ohne pyplot global zu veraendern."""
    def __enter__(self):
        import matplotlib as mpl
        style = dict(STYLE)
        for k in ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
                  "ytick.labelsize", "legend.fontsize", "figure.titlesize",
                  "legend.title_fontsize"):
            style[k] = FONT_PT
        self._ctx = mpl.rc_context(style)
        self._ctx.__enter__()

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
