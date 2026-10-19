"""FlatTop-Auswertung: 2D-Supergauss-Fit und Kennzahlen.

Hier wird NUR gerechnet. Das Ergebnis ist ein FlatTopResult, aus dem
plotting.py die Abbildungen und export.py die Dateien erzeugt.

Ablauf von analyze_image():
  1. Hintergrund grob aus dem Bildrand schaetzen, Strahl finden
     (groesste zusammenhaengende Flaeche ueber 50 %)
  2. ROI um den Strahl ausschneiden, fuer den Fit ggf. binnen (Tempo)
  3. rotierten 2D-Supergauss fitten (Offset B ist Fitparameter)
  4. Kennzahlen:
       - Halbachsen bei 1/e^2, 50 %, 90 % (aus dem Fit)
       - Kantenbreite 10-90 % entlang langer/kurzer Achse
       - Aequivalent-Durchmesser der 50 %-Flaeche (direkt aus den Daten)
       - Plateau-Uniformity (std/mean, Peak-to-Valley) in der 90 %-Ellipse
       - Flatness-Kurve: Uniformity in Kreisen wachsenden Durchmessers
       - Schnitte horizontal / vertikal / lange / kurze Achse
       - Uebersteuerung
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

from .imageio import is_simulation_file, load_image, load_simulation
from .models import (
    supergaussian_2d,
    radius_at_level,
    ellipse_radius_along,
    normalize_axes,
)

LEVEL_E2 = math.exp(-2)


# ---------------------------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------------------------
@dataclass
class AnalysisSettings:
    # Standard: IDS-Kamera (Sony-Sensor, 1.45 um Pixel) mit 40er Mikroskopobjektiv
    # -> 1.45 / 43.75 = 0.0331 um pro Pixel in der Objektebene
    pixel_size_um: float = 1.45        # Kamerapixel in um
    magnification: float = 43.75       # Abbildungsmassstab (Objekt -> Kamera)
    background_path: str | None = None # optionales Dunkel-/Hintergrundbild
    roi_factor: float = 2.5            # ROI-Halbbreite = roi_factor * r50
    fit_max_pixels: int = 250          # ROI wird fuer den Fit auf max. so viele px gebinnt
    m_max: float = 20.0                # Obergrenze Supergauss-Ordnung
    plateau_level: float = 0.9         # Level, das das Plateau definiert
    plateau_scale: float = 1.0         # Plateau-Ellipse = scale * r(plateau_level)
    flatness_step_px: int = 2          # Schrittweite Kreisdurchmesser (Kamerapixel)
    cut_halfwidth_um: float = 0.0      # 0 = automatisch
    levels: tuple = (LEVEL_E2, 0.5, 0.9)
    edge_levels: tuple = (0.1, 0.9)

    # --- Abschaetzung Uniformity/Crosstalk im Fallengitter -----------------
    # Definitionen wie in Multitone_Simulation (Combinated_Optimization/lib/
    # single_beam.py, weighted_multitone_flattop_optimizer.py); Standardwerte
    # von dort (Rb-85, T = 17 uK, nu_r = 60.4 kHz, Pitch 5.288 um).
    lattice_estimate: bool = True      # abschaltbar (z. B. fuer grosse Profile)
    pitch_um: float = 5.288            # Abstand der Fallen (Quadratgitter)
    neighbour_ring: int = 1            # 1 = die 8 direkten Nachbarn
    atom_temperature_uK: float = 17.0
    trap_freq_kHz: float = 60.4        # radiale Fallenfrequenz nu_r
    atom_mass_u: float = 84.911789738  # Rb-85
    weighted_n_sigma: float = 6.0      # lokales Gitter +- n * sigma_atom
    weighted_n_grid: int = 241
    uniformity_threshold: float = 0.05  # Schwelle fuer "Uniformity-Radius" (5 %)
    min_coverage: float = 0.95         # Anteil der Nachbar-Stuetzpunkte, die im Bild liegen muessen

    @property
    def um_per_px(self) -> float:
        return self.pixel_size_um / self.magnification


# ---------------------------------------------------------------------------
# Ergebnis-Container
# ---------------------------------------------------------------------------
@dataclass
class SupergaussFit:
    A: float
    x0: float          # px (volle Aufloesung, Bildkoordinaten)
    y0: float
    wa: float          # px, lange Halbachse-Parameter
    wb: float          # px, kurze
    theta: float       # rad
    m: float
    B: float
    errors: dict = field(default_factory=dict)
    r_squared: float = float("nan")
    binning: int = 1

    def params(self):
        return (self.A, self.x0, self.y0, self.wa, self.wb, self.theta, self.m, self.B)


@dataclass
class CrossSection:
    name: str
    angle: float              # rad, Richtung des Schnitts in Bildkoordinaten
    pos_um: np.ndarray
    data: np.ndarray          # (I - B) / A
    model: np.ndarray         # Fit, gleich normiert
    level_radii_um: dict      # label -> Radius, an dem der Schnitt den Level kreuzt


@dataclass
class FlatTopResult:
    path: str
    settings: AnalysisSettings
    image_shape: tuple
    saturation_value: float
    fit: SupergaussFit
    roi: tuple                         # (y0, y1, x0, x1) in px
    roi_data: np.ndarray               # Rohdaten in der ROI (nach Hintergrundabzug)
    roi_model: np.ndarray              # Fit in der ROI
    view: tuple                        # (y0, y1, x0, x1) Ausschnitt fuer die 2D-Darstellung
    view_data: np.ndarray              # inkl. Nebenstruktur (Airy-Ringe), s. signal_extent_px
    signal_extent_px: float            # Radius, bis zu dem Signal ueber dem Untergrund liegt
    radii_px: dict                     # label -> (a, b) Halbachsen in px
    edge_width_um: tuple               # (lange, kurze Achse) 10-90 %
    d_eq50_um: float                   # Aequivalent-Durchmesser 50 %-Flaeche (Daten)
    plateau_uniformity: float          # std/mean in der Plateau-Ellipse
    plateau_pv: float                  # (max-min)/mean in der Plateau-Ellipse
    plateau_npix: int
    region_stats: dict                 # Level -> {"U": sigma/mu, "PV": (max-min)/mu, "npix"} in der Level-Ellipse
    flatness_d_um: np.ndarray
    flatness_u: np.ndarray
    cuts: dict                         # name -> CrossSection
    max_value: float
    saturated_fraction: float          # Anteil uebersteuerter Pixel in der 50 %-Flaeche
    warnings: list = field(default_factory=list)
    lattice: dict | None = None        # Abschaetzung im Fallengitter, s. lattice_estimate()
    background: float = 0.0            # Untergrund (Bildrand-Mittel), Counts
    background_err: float = 0.0        # dessen Unsicherheit (Rand-Ungleichmaessigkeit)
    simulation: dict | None = None     # Parameter, falls simuliertes Profil (.npz), sonst None

    # -- Hilfen ------------------------------------------------------------
    @property
    def name(self) -> str:
        return Path(self.path).stem

    @property
    def is_simulation(self) -> bool:
        return self.simulation is not None

    @property
    def um_per_px(self) -> float:
        return self.settings.um_per_px

    @property
    def uniformity_radius_um(self) -> float:
        """Radius des groessten Kreises um das Zentrum, in dem die Uniformity
        sigma/mu noch unter der Schwelle (Standard 5 %) liegt.

        Die Flatness-Kurve waechst mit dem Durchmesser; genommen wird der erste
        Schnittpunkt mit der Schwelle, linear zwischen den Stuetzstellen
        interpoliert. NaN, wenn die Schwelle gar nicht erreicht wird (dann ist
        das Profil bis zum Rand besser als die Schwelle) oder schon der kleinste
        Kreis darueber liegt.
        """
        d, u = self.flatness_d_um, self.flatness_u
        thr = self.settings.uniformity_threshold
        if d.size < 2 or u[0] >= thr or np.nanmax(u) < thr:
            return float("nan")
        i = int(np.argmax(u >= thr))        # erster Punkt ueber der Schwelle
        u0, u1, d0, d1 = u[i - 1], u[i], d[i - 1], d[i]
        d_cross = d0 + (thr - u0) * (d1 - d0) / (u1 - u0) if u1 > u0 else d0
        return float(d_cross / 2)

    def radii_hv_um(self) -> dict:
        """Level -> (horizontal, vertikal): Radius entlang der Bild-x/y-Achse (um)."""
        h, v = self.cuts["horizontal"], self.cuts["vertikal"]
        return {k: (h.level_radii_um[k], v.level_radii_um[k]) for k in self.radii_px}

    def radii_um(self) -> dict:
        s = self.um_per_px
        return {k: (a * s, b * s) for k, (a, b) in self.radii_px.items()}

    def summary(self) -> dict:
        """Flache Kennzahlen-Tabelle (fuer GUI, JSON und Batch-CSV)."""
        s = self.um_per_px
        f = self.fit
        e = f.errors
        row = {
            "file": Path(self.path).name,
            "um_per_px": s,
            "x0_um": f.x0 * s,
            "y0_um": f.y0 * s,
            "theta_deg": math.degrees(f.theta),
            "m": f.m,
            "m_err": e.get("m", float("nan")),
            "A": f.A,
            "B": f.B,
            "R2": f.r_squared,
        }
        for label, (a, b) in self.radii_um().items():
            row[f"D_long_{label}_um"] = 2 * a
            row[f"D_short_{label}_um"] = 2 * b
        row["ellipticity_e2"] = self.radii_px[_label(LEVEL_E2)][1] / self.radii_px[_label(LEVEL_E2)][0]
        row["edge_long_um"] = self.edge_width_um[0]
        row["edge_short_um"] = self.edge_width_um[1]
        row["D_eq50_data_um"] = self.d_eq50_um
        row["D_signal_um"] = 2 * self.signal_extent_px * s
        row["r_uniform_um"] = self.uniformity_radius_um
        row["uniformity_threshold_pct"] = 100 * self.settings.uniformity_threshold
        for label, (a, b) in self.radii_um().items():
            rh, rv = self.radii_hv_um()[label]
            row[f"w_long_{label}_um"] = a
            row[f"w_short_{label}_um"] = b
            row[f"w_h_{label}_um"] = rh
            row[f"w_v_{label}_um"] = rv
            st = self.region_stats.get(label, {})
            row[f"U_{label}_pct"] = 100 * st.get("U", float("nan"))
            row[f"PV_{label}_pct"] = 100 * st.get("PV", float("nan"))
        if self.lattice:
            L = self.lattice
            for label in self.radii_px:
                row[f"eta_{label}_pct"] = 100 * L["hard"][label]["eta"]
            row["U_atom_pct"] = 100 * L["U_w"]
            row["eta_atom_pct"] = 100 * L["eta_w"]
            row["sigma_atom_nm"] = 1e3 * L["sigma_atom_um"]
            row["pitch_um"] = L["pitch_um"]
        row["U_plateau_rms_pct"] = 100 * self.plateau_uniformity
        row["PV_plateau_pct"] = 100 * self.plateau_pv
        row["plateau_npix"] = self.plateau_npix
        row["max_value"] = self.max_value
        row["saturated_pct"] = 100 * self.saturated_fraction
        row["warnings"] = " | ".join(self.warnings)
        return row


def _label(level: float) -> str:
    if abs(level - LEVEL_E2) < 1e-9:
        return "1/e2"
    return f"{round(level * 100):d}%"


def level_labels(settings: AnalysisSettings) -> list[str]:
    return [_label(l) for l in settings.levels]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _border_median(img: np.ndarray, frac: float = 0.05) -> float:
    ny, nx = img.shape
    by, bx = max(1, int(ny * frac)), max(1, int(nx * frac))
    border = np.concatenate([
        img[:by].ravel(), img[-by:].ravel(), img[:, :bx].ravel(), img[:, -bx:].ravel()
    ])
    return float(np.median(border))


def border_background(img: np.ndarray, frac: float = 0.05) -> tuple[float, float]:
    """Untergrund aus den vier Bildraendern: (Mittelwert, Unsicherheit).

    Mittelwert statt Median, weil 8-bit-Bilder nahe 0 nur ganze Zahlen haben
    und der Median dann um bis zu 1 Count springt. Ausreisser (Hotpixel,
    angeschnittene Strukturen) werden vorher abgeschnitten.
    Unsicherheit = halbe Spannweite der vier Rand-Mittelwerte (ungleichmaessiger
    Untergrund), mindestens 0.05 Counts.
    """
    ny, nx = img.shape
    by, bx = max(1, int(ny * frac)), max(1, int(nx * frac))
    strips = [img[:by], img[-by:], img[:, :bx], img[:, -bx:]]
    allv = np.concatenate([s_.ravel() for s_ in strips])
    med = np.median(allv)
    mad = np.median(np.abs(allv - med)) * 1.4826 + 0.5
    lim = med + 5 * mad
    means = [float(s_[s_ <= lim].mean()) if np.any(s_ <= lim) else float(med) for s_ in strips]
    bg = float(allv[allv <= lim].mean())
    dbg = max(0.5 * (max(means) - min(means)), 0.05)
    return bg, dbg


def _main_blob(img: np.ndarray, bg: float, rel: float = 0.5) -> np.ndarray:
    """Maske der groessten zusammenhaengenden Flaeche ueber bg + rel*(max-bg).

    Das Maximum wird auf einem geglaetteten Bild bestimmt, damit einzelne
    heisse Pixel die Schwelle nicht verschieben.
    """
    smooth = ndimage.uniform_filter(img, size=5)
    peak = np.percentile(smooth, 99.9)
    mask = smooth > bg + rel * (peak - bg)
    lab, n = ndimage.label(mask)
    if n == 0:
        raise RuntimeError("Kein Strahl gefunden (Bild leer oder nur Rauschen?)")
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def _moments(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    xc, yc = xs.mean(), ys.mean()
    cxx = ((xs - xc) ** 2).mean()
    cyy = ((ys - yc) ** 2).mean()
    cxy = ((xs - xc) * (ys - yc)).mean()
    theta = 0.5 * math.atan2(2 * cxy, cxx - cyy)
    ev = np.linalg.eigvalsh([[cxx, cxy], [cxy, cyy]])
    # gleichmaessig gefuellte Ellipse: Varianz = r^2/4
    rb, ra = 2 * np.sqrt(np.maximum(ev, 1e-12))
    return xc, yc, ra, rb, theta


def _robust_pv(v: np.ndarray) -> float:
    """Peak-to-Valley relativ zum Mittelwert, aus dem 0,5- und 99,5-%-Perzentil.

    Einzelne Staubkoerner/Hotpixel sollen den Wert nicht bestimmen.
    """
    lo, hi = np.percentile(v, [0.5, 99.5])
    return float((hi - lo) / v.mean())


def _bin(img: np.ndarray, b: int) -> np.ndarray:
    if b <= 1:
        return img
    ny, nx = (img.shape[0] // b) * b, (img.shape[1] // b) * b
    return img[:ny, :nx].reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_supergaussian(img: np.ndarray, settings: AnalysisSettings):
    """Fittet den 2D-Supergauss in einer ROI um den Strahl.

    Returns (SupergaussFit, roi, blob50_mask)
    """
    ny, nx = img.shape
    bg0 = _border_median(img)
    blob = _main_blob(img, bg0, 0.5)
    xc, yc, ra, rb, th = _moments(blob)

    half = settings.roi_factor * ra
    y0r, y1r = int(max(0, yc - half)), int(min(ny, yc + half + 1))
    x0r, x1r = int(max(0, xc - half)), int(min(nx, xc + half + 1))
    roi_img = img[y0r:y1r, x0r:x1r]

    b = max(1, int(math.ceil(max(roi_img.shape) / settings.fit_max_pixels)))
    fit_img = _bin(roi_img, b)
    fy, fx = fit_img.shape
    # Mittelpunkte der Bin-Bloecke in voller Aufloesung (Pixelzentren bei ganzen Zahlen)
    X, Y = np.meshgrid(x0r + np.arange(fx) * b + (b - 1) / 2,
                       y0r + np.arange(fy) * b + (b - 1) / 2)

    A0 = float(np.percentile(roi_img[blob[y0r:y1r, x0r:x1r]], 50) - bg0) \
        if blob[y0r:y1r, x0r:x1r].any() else float(roi_img.max() - bg0)
    m0 = 3.0
    # r50 = w * ln2^(1/2m)  ->  w = r50 / ln2^(1/2m)
    s50 = math.log(2) ** (1 / (2 * m0))
    p0 = [max(A0, 1e-6), xc, yc, ra / s50, rb / s50, th, m0, bg0]
    amax = float(img.max())
    lo = [0, x0r, y0r, 0.5, 0.5, -np.pi, 0.5, -amax]
    hi = [2 * max(amax, 1.0), x1r, y1r, 2 * half + 1, 2 * half + 1, np.pi, settings.m_max, amax]
    p0 = [min(max(v, l + 1e-9), h - 1e-9) for v, l, h in zip(p0, lo, hi)]

    popt, pcov = curve_fit(
        supergaussian_2d, (X.ravel(), Y.ravel()), fit_img.ravel(),
        p0=p0, bounds=(lo, hi), maxfev=20000,
    )
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    A, x0, y0, wa, wb, theta, m, B = popt
    wa, wb, theta, swa, swb = normalize_axes(wa, wb, theta, perr[3], perr[4])

    fit = SupergaussFit(
        A=A, x0=x0, y0=y0, wa=wa, wb=wb, theta=theta, m=m, B=B,
        errors={"A": perr[0], "x0": perr[1], "y0": perr[2], "wa": swa, "wb": swb,
                "theta": perr[5], "m": perr[6], "B": perr[7]},
        binning=b,
    )
    at_bound = [n for n, v, l, h in zip("A x0 y0 wa wb theta m B".split(), popt, lo, hi)
                if np.isclose(v, l) or np.isclose(v, h)]
    return fit, (y0r, y1r, x0r, x1r), blob, at_bound


# ---------------------------------------------------------------------------
# Kennzahlen
# ---------------------------------------------------------------------------
def signal_extent(img: np.ndarray, fit: SupergaussFit, noise: float, bg: float) -> float:
    """Radius (px), bis zu dem das azimutal gemittelte Signal ueber dem Untergrund liegt.

    Damit werden Nebenstrukturen (z. B. Airy-Ringe) mit in den Plotausschnitt
    genommen, auch wenn der Supergauss-Fit sie nicht beschreibt.
    Schwelle: Untergrund (Bildrand-Median, NICHT der Fit-Offset B - der
    schluckt bei Airy-Profilen die Ringe) + max(0,5 % von A, 3 sigma des Ringmittels).
    Dunkle Ringe dazwischen beenden die Suche nicht: genommen wird der
    aeusserste Radius ueber der Schwelle.
    """
    ny, nx = img.shape
    Y, X = np.indices(img.shape)
    r = np.hypot(X - fit.x0, Y - fit.y0).astype(int).ravel()
    cnt = np.bincount(r)
    prof = np.bincount(r, img.ravel()) / np.maximum(cnt, 1)
    k = 9
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    thr = bg + np.maximum(0.005 * fit.A, 3 * noise / np.sqrt(np.maximum(cnt, 1) * k))
    # nur Radien, deren Ring noch ueberwiegend im Bild liegt
    r_max = int(min(max(fit.x0, nx - fit.x0), max(fit.y0, ny - fit.y0)))
    above = np.nonzero(prof[:r_max] > thr[:r_max])[0]
    if above.size == 0:
        return 0.0
    return float(above[-1])


def _cross_section(img, fit: SupergaussFit, phi: float, half_px: float, name: str,
                   radii_px: dict, um_per_px: float) -> CrossSection:
    n = int(max(200, 4 * half_px))
    t = np.linspace(-half_px, half_px, n)
    xs = fit.x0 + t * math.cos(phi)
    ys = fit.y0 + t * math.sin(phi)
    raw = ndimage.map_coordinates(img, [ys, xs], order=1, mode="nearest")
    model = supergaussian_2d((xs, ys), *fit.params())
    norm = lambda v: (v - fit.B) / fit.A
    lr = {lab: ellipse_radius_along(a, b, fit.theta, phi) * um_per_px
          for lab, (a, b) in radii_px.items()}
    return CrossSection(name, phi, t * um_per_px, norm(raw), norm(model), lr)


def analyze_image(img: np.ndarray, settings: AnalysisSettings | None = None,
                  path: str = "", saturation_value: float = 255.0) -> FlatTopResult:
    settings = settings or AnalysisSettings()
    s = settings.um_per_px
    warnings: list[str] = []

    fit, roi, blob50, at_bound = fit_supergaussian(img, settings)
    if at_bound:
        warnings.append("Fitparameter an der Grenze: " + ", ".join(at_bound))

    y0r, y1r, x0r, x1r = roi
    roi_data = img[y0r:y1r, x0r:x1r]
    Yr, Xr = np.mgrid[y0r:y1r, x0r:x1r]
    roi_model = supergaussian_2d((Xr, Yr), *fit.params())
    ss_res = np.sum((roi_data - roi_model) ** 2)
    ss_tot = np.sum((roi_data - roi_data.mean()) ** 2)
    fit.r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Halbachsen bei den Levels
    radii_px = {_label(l): (radius_at_level(fit.wa, fit.m, l), radius_at_level(fit.wb, fit.m, l))
                for l in settings.levels}
    lo, hi = settings.edge_levels
    edge = ((radius_at_level(fit.wa, fit.m, lo) - radius_at_level(fit.wa, fit.m, hi)) * s,
            (radius_at_level(fit.wb, fit.m, lo) - radius_at_level(fit.wb, fit.m, hi)) * s)

    a_e2, b_e2 = radii_px[_label(LEVEL_E2)]
    ny, nx = img.shape
    if (fit.x0 - a_e2 < 0 or fit.y0 - a_e2 < 0 or fit.x0 + a_e2 > nx or fit.y0 + a_e2 > ny):
        warnings.append("1/e²-Ellipse reicht über den Bildrand")

    # Aequivalent-Durchmesser der 50 %-Flaeche, direkt aus den Daten (Schwelle aus dem Fit)
    thr50 = fit.B + 0.5 * fit.A
    lab, nlab = ndimage.label(ndimage.uniform_filter(img, 3) > thr50)
    if nlab:
        idx = lab[int(round(np.clip(fit.y0, 0, ny - 1))), int(round(np.clip(fit.x0, 0, nx - 1)))]
        if idx == 0:
            idx = int(np.argmax(ndimage.sum(np.ones_like(img), lab, range(1, nlab + 1)))) + 1
        area50 = float((lab == idx).sum())
    else:
        area50 = 0.0
    d_eq50 = 2 * math.sqrt(area50 / math.pi) * s

    # Hintergrundkorrigierte Daten fuer Uniformity
    # Untergrund fuer Uniformity/Crosstalk: Median am Bildrand. NICHT der
    # Fit-Offset B - der liegt bei Profilen mit Nebenstruktur (Airy-Ringe,
    # Halo) zu hoch, dann wuerden die Fluegel negativ und der Crosstalk falsch.
    bg, dbg = border_background(img)
    corr = roi_data - bg
    ct, st = math.cos(fit.theta), math.sin(fit.theta)
    dx, dy = Xr - fit.x0, Yr - fit.y0
    xp, yp = dx * ct + dy * st, -dx * st + dy * ct

    pa = settings.plateau_scale * radius_at_level(fit.wa, fit.m, settings.plateau_level)
    pb = settings.plateau_scale * radius_at_level(fit.wb, fit.m, settings.plateau_level)
    pmask = (xp / pa) ** 2 + (yp / pb) ** 2 <= 1
    if pmask.sum() >= 5:
        vals = corr[pmask]
        mean = vals.mean()
        u_rms = float(vals.std() / mean)
        u_pv = _robust_pv(vals)
    else:
        u_rms = u_pv = float("nan")
        warnings.append("Plateau-Ellipse enthält < 5 Pixel")

    # Uniformity in den drei Regionen (Level-Ellipsen 1/e^2, 50 %, 90 %), volles Bild
    region_stats = {}
    for lab, (ra, rb) in radii_px.items():
        yy0, yy1 = int(max(0, fit.y0 - ra - 1)), int(min(ny, fit.y0 + ra + 2))
        xx0, xx1 = int(max(0, fit.x0 - ra - 1)), int(min(nx, fit.x0 + ra + 2))
        Yb, Xb = np.mgrid[yy0:yy1, xx0:xx1]
        ddx, ddy = Xb - fit.x0, Yb - fit.y0
        xq, yq = ddx * ct + ddy * st, -ddx * st + ddy * ct
        m = (xq / ra) ** 2 + (yq / rb) ** 2 <= 1
        v = img[yy0:yy1, xx0:xx1][m] - bg
        if v.size >= 5 and v.mean() > 0:
            region_stats[lab] = {"U": float(v.std() / v.mean()),
                                 "PV": _robust_pv(v),
                                 "npix": int(v.size)}
        else:
            region_stats[lab] = {"U": float("nan"), "PV": float("nan"), "npix": int(v.size)}

    # Abschaetzung im Fallengitter (optional)
    lattice = None
    if settings.lattice_estimate:
        lattice = lattice_estimate(img, fit, radii_px, settings)
        if lattice["too_small"]:
            warnings.append(f"Crosstalk nicht bestimmbar: Bild reicht nicht bis zu den Nachbarn "
                            f"(Pitch {settings.pitch_um:g} µm, nur "
                            f"{100 * lattice['coverage']:.0f} % im Bild)")
        elif lattice["truncated"]:
            warnings.append("Crosstalk-Abschätzung: Nachbarn teilweise außerhalb des Bildes "
                            "(dort als 0 gerechnet)")
        if lattice["sigma_atom_px"] < 1.5:
            warnings.append("σ_atom ist kleiner als 1,5 Pixel - gewichtete Werte ungenau")

    # Flatness-Kurve: Kreise um das Fitzentrum. Einmal nach Radius sortieren und
    # kumulierte Summen nutzen -> alle Kreise in einem Durchgang (statt je Kreis
    # das ganze ROI-Array zu maskieren).
    rr = np.hypot(dx, dy).ravel()
    order = np.argsort(rr)
    r_sorted = rr[order]
    v_sorted = corr.ravel()[order].astype(np.float64)
    cs1 = np.cumsum(v_sorted)
    cs2 = np.cumsum(v_sorted ** 2)
    step = max(1, int(settings.flatness_step_px))
    d_max = int(2 * b_e2)
    ds = np.arange(step, max(d_max, step) + 1, step)
    n_in = np.searchsorted(r_sorted, ds / 2, side="right")
    d_list, u_list = [], []
    for d, n in zip(ds, n_in):
        if n < 3:
            continue
        mean = cs1[n - 1] / n
        if mean <= 0:
            continue
        var = max(cs2[n - 1] / n - mean ** 2, 0.0)
        d_list.append(d * s)
        u_list.append(float(np.sqrt(var) / mean))

    # Uebersteuerung
    max_val = float(img.max())
    sat = float(np.mean(img[blob50] >= saturation_value)) if blob50.any() else 0.0
    if sat > 1e-3:
        warnings.append(f"{100 * sat:.1f} % der Pixel in der 50 %-Fläche übersteuert")
    if fit.A < 20 * np.std(img[:max(1, ny // 20)]):
        warnings.append("geringes Signal-Rausch-Verhaeltnis")

    # Schnitte
    # Plotausschnitt: mindestens 1.6 * r(1/e^2), aber so gross, dass Nebenstrukturen
    # (Airy-Ringe etc.) nicht abgeschnitten werden
    noise = float(np.std(img[:max(1, ny // 20)]))
    r_sig = signal_extent(img, fit, noise, _border_median(img))
    if settings.cut_halfwidth_um > 0:
        half_px = settings.cut_halfwidth_um / s
    else:
        half_px = max(1.6 * a_e2, 1.15 * r_sig)
    # Rand: der 15 %-Zuschlag darf wegfallen, die Nebenstruktur selbst nicht
    d_min = min(fit.x0, nx - fit.x0, fit.y0, ny - fit.y0)
    if settings.cut_halfwidth_um <= 0 and half_px > d_min:
        half_px = max(d_min, r_sig, 1.6 * a_e2)
    vy0, vy1 = int(max(0, np.floor(fit.y0 - half_px))), int(min(ny, np.ceil(fit.y0 + half_px) + 1))
    vx0, vx1 = int(max(0, np.floor(fit.x0 - half_px))), int(min(nx, np.ceil(fit.x0 + half_px) + 1))
    if r_sig > 0 and (fit.x0 - r_sig < 0 or fit.y0 - r_sig < 0
                      or fit.x0 + r_sig > nx or fit.y0 + r_sig > ny):
        warnings.append("Nebenstruktur reicht bis an den Bildrand")
    cuts = {}
    for name, phi in (("horizontal", 0.0), ("vertikal", math.pi / 2),
                      ("lange Achse", fit.theta), ("kurze Achse", fit.theta + math.pi / 2)):
        cuts[name] = _cross_section(img, fit, phi, half_px, name, radii_px, s)

    return FlatTopResult(
        path=str(path), settings=settings, image_shape=img.shape,
        saturation_value=saturation_value, fit=fit, roi=roi,
        roi_data=roi_data, roi_model=roi_model,
        view=(vy0, vy1, vx0, vx1), view_data=img[vy0:vy1, vx0:vx1], signal_extent_px=r_sig,
        radii_px=radii_px,
        edge_width_um=edge, d_eq50_um=d_eq50,
        plateau_uniformity=u_rms, plateau_pv=u_pv, plateau_npix=int(pmask.sum()),
        region_stats=region_stats,
        lattice=lattice, background=bg, background_err=dbg,
        flatness_d_um=np.asarray(d_list), flatness_u=np.asarray(u_list),
        cuts=cuts, max_value=max_val, saturated_fraction=sat, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Abschaetzung im Fallengitter (Uniformity und Crosstalk, hart + atomgewichtet)
# ---------------------------------------------------------------------------
def sigma_thermal_um(settings: AnalysisSettings) -> float:
    """Thermische 1-sigma-Breite der Atom-Ortsverteilung (um), inkl. Nullpunkt:
    sigma^2 = hbar/(2 m omega) * coth(hbar omega / (2 kB T))  - wie im Optimierer."""
    from scipy.constants import hbar, k as kB, atomic_mass
    m = settings.atom_mass_u * atomic_mass
    omega = 2 * np.pi * settings.trap_freq_kHz * 1e3
    T = settings.atom_temperature_uK * 1e-6
    if omega <= 0 or T <= 0:
        return float("nan")
    x = hbar * omega / (2 * kB * T)
    return float(np.sqrt(hbar / (2 * m * omega) / np.tanh(x)) * 1e6)


def _neighbour_offsets_px(pitch_px: float, ring: int):
    idx = np.arange(-ring, ring + 1)
    ix, iy = np.meshgrid(idx, idx)
    keep = ~((ix == 0) & (iy == 0))
    return ix[keep].ravel() * pitch_px, iy[keep].ravel() * pitch_px


def lattice_estimate(img: np.ndarray, fit: SupergaussFit, radii_px: dict,
                     settings: AnalysisSettings) -> dict:
    """Uniformity und Crosstalk des GEMESSENEN Profils, wenn es auf einem
    Quadratgitter (Pitch) an jeder Site steht - Definitionen aus
    Multitone_Simulation (single_beam.py):

        I_own(r)  = gemessenes Profil (Untergrund = Bildrand-Median abgezogen),
                    Site im Fitzentrum
        I_nb(r)   = Summe desselben Profils, um +-pitch verschoben (8 Nachbarn)

      hart, je Region R (Level-Ellipse 1/e^2, 50 %, 90 %):
        U_h   = std(I_own) / mean(I_own)          ueber R
        eta_h = sum(I_nb) / sum(I_own)            ueber R
      atomgewichtet, W = exp(-r^2 / (2 sigma_atom^2)) um die Site:
        U_w   = sqrt(<(I-<I>_W)^2>_W) / <I>_W
        eta_w = sum(I_nb * W) / sum(I_own * W)

    Das Profil wird bilinear aus dem Kamerabild interpoliert; Punkte ausserhalb
    des Bildes zaehlen als 0 (-> Warnung, falls Nachbarbeitraege abgeschnitten).
    """
    s = settings.um_per_px
    ny, nx = img.shape
    bg, dbg = border_background(img)
    I0 = img - bg
    pitch_px = settings.pitch_um / s
    ox, oy = _neighbour_offsets_px(pitch_px, int(settings.neighbour_ring))
    ct, st = math.cos(fit.theta), math.sin(fit.theta)

    def sample(X, Y):
        return ndimage.map_coordinates(I0, [Y.ravel(), X.ravel()], order=1,
                                       mode="constant", cval=0.0).reshape(X.shape)

    def own_and_nb(X, Y):
        """X, Y: Positionen relativ zur Site (px). Liefert zusaetzlich den
        Anteil der Nachbar-Stuetzpunkte, die innerhalb des Kamerabilds liegen."""
        own = sample(fit.x0 + X, fit.y0 + Y)
        nb = np.zeros_like(own)
        n_in = 0
        n_all = 0
        for dx, dy in zip(ox, oy):
            # Nachbar-Site bei (dx, dy): sein Licht am Ort r ist I(r - (dx, dy))
            xx, yy = fit.x0 + X - dx, fit.y0 + Y - dy
            nb += sample(xx, yy)
            n_in += np.count_nonzero((xx >= 0) & (xx <= nx - 1) & (yy >= 0) & (yy <= ny - 1))
            n_all += xx.size
        return own, nb, n_in / max(n_all, 1)

    hard = {}
    coverages = []
    for lab, (ra, rb) in radii_px.items():
        n = int(max(41, min(401, 2 * ra + 1)))
        xs = (np.arange(n) + 0.5) / n * 2 * ra - ra          # Zellmitten
        X, Y = np.meshgrid(xs, xs)
        xp, yp = X * ct + Y * st, -X * st + Y * ct
        inside = (xp / ra) ** 2 + (yp / rb) ** 2 <= 1
        own, nb, cov = own_and_nb(X, Y)
        coverages.append(cov)
        o, b = own[inside], nb[inside]
        mean = o.mean()
        eta = float(b.sum() / o.sum()) if o.sum() > 0 else float("nan")
        ok = cov >= settings.min_coverage
        hard[lab] = {
            "U": float(o.std() / mean) if mean > 0 else float("nan"),
            # reicht das Bild nicht bis zu den Nachbarn, ist eta nicht bestimmbar
            "eta": eta if ok else float("nan"),
            # Unsicherheit aus dem Untergrund: jeder Nachbar traegt dB bei
            "eta_err": float(len(ox) * dbg / mean) if (ok and mean > 0) else float("nan"),
            "coverage": float(cov),
        }

    sig_um = sigma_thermal_um(settings)
    sig_px = sig_um / s
    half = settings.weighted_n_sigma * sig_px
    xs = np.linspace(-half, half, int(settings.weighted_n_grid))
    X, Y = np.meshgrid(xs, xs)
    own, nb, cov_w = own_and_nb(X, Y)
    coverages.append(cov_w)
    W = np.exp(-(X ** 2 + Y ** 2) / (2 * sig_px ** 2))
    muw = np.sum(own * W) / np.sum(W)
    U_w = float(np.sqrt(np.sum((own - muw) ** 2 * W) / np.sum(W)) / muw) if muw > 0 else float("nan")
    eta_w = float(np.sum(nb * W) / np.sum(own * W)) if np.sum(own * W) > 0 else float("nan")
    eta_w_err = float(len(ox) * dbg / muw) if muw > 0 else float("nan")
    if cov_w < settings.min_coverage:
        eta_w = eta_w_err = float("nan")
    truncated = min(coverages) < 1.0
    too_small = min(coverages) < settings.min_coverage

    return {
        "pitch_um": settings.pitch_um,
        "neighbour_ring": int(settings.neighbour_ring),
        "sigma_atom_um": sig_um,
        "atom": {"T_uK": settings.atom_temperature_uK, "nu_kHz": settings.trap_freq_kHz,
                 "mass_u": settings.atom_mass_u},
        "hard": hard,
        "U_w": U_w,
        "eta_w": eta_w,
        "eta_w_err": eta_w_err,
        "background": bg,
        "background_err": dbg,
        "sigma_atom_px": sig_px,
        "truncated": truncated,
        "too_small": too_small,
        "coverage": float(min(coverages)),
    }


_BG_CACHE: dict = {}


def load_background(path: str | None):
    if not path:
        return None
    if path not in _BG_CACHE:
        _BG_CACHE[path] = load_image(path)[0]
    return _BG_CACHE[path]


def analyze_simulation(path: str | Path, settings: AnalysisSettings | None = None) -> FlatTopResult:
    """Simuliertes 2D-Profil (.npz) mit denselben Einstellungen wie ein Messbild.

    Pixelgroesse kommt aus der Datei (pixel_size_um = um pro Pixel, Vergroesserung
    1), kein Hintergrundabzug, keine Uebersteuerung. Intensitaet: Maximum = 1.
    """
    from dataclasses import replace
    sim = load_simulation(path)
    s_sim = replace(settings or AnalysisSettings(), pixel_size_um=sim["um_per_px"],
                    magnification=1.0, background_path=None)
    res = analyze_image(sim["image"], s_sim, str(path), sim["saturation"])
    res.saturated_fraction = 0.0
    res.warnings = [w for w in res.warnings
                    if "übersteuert" not in w and "Rausch" not in w]
    res.simulation = dict(sim["params"]) or {"source": "Simulation"}
    return res


def analyze_file(path: str | Path, settings: AnalysisSettings | None = None) -> FlatTopResult:
    if is_simulation_file(path):
        return analyze_simulation(path, settings)
    settings = settings or AnalysisSettings()
    img, sat = load_image(path)
    raw_max = img.max()
    bg = load_background(settings.background_path)
    if bg is not None:
        if bg.shape != img.shape:
            raise ValueError(f"Hintergrundbild {bg.shape} passt nicht zu {img.shape}")
        img = img - bg
        raw = img + bg
    else:
        raw = img
    res = analyze_image(img, settings, str(path), sat)
    # Uebersteuerung immer am Rohbild beurteilen
    res.max_value = float(raw_max)
    if bg is not None:
        f = res.fit
        m50 = img > f.B + 0.5 * f.A
        res.saturated_fraction = float(np.mean(raw[m50] >= sat)) if m50.any() else 0.0
    return res
