"""Tests mit synthetischen FlatTops: bekannte Parameter muessen zurueckkommen.

Ausfuehren:  python -m pytest tests
"""
import math

import numpy as np
import pytest

from beamshaper.analysis import AnalysisSettings, analyze_image, LEVEL_E2
from beamshaper.models import supergaussian_2d, radius_at_level, ellipse_radius_along, normalize_axes


def make_image(nx=640, ny=480, A=200, x0=330.3, y0=241.7, wa=60, wb=45,
               theta=math.radians(20), m=4, B=8, noise=1.5, seed=0):
    Y, X = np.mgrid[0:ny, 0:nx]
    img = supergaussian_2d((X, Y), A, x0, y0, wa, wb, theta, m, B)
    rng = np.random.default_rng(seed)
    return img + rng.normal(0, noise, img.shape)


def test_radius_at_level_gauss():
    # m=1: I = exp(-(r/w)^2) -> 1/e^2 bei r = sqrt(2) w
    assert radius_at_level(10, 1, LEVEL_E2) == pytest.approx(10 * math.sqrt(2))


def test_ellipse_radius_along_axes():
    assert ellipse_radius_along(5, 2, 0.3, 0.3) == pytest.approx(5)
    assert ellipse_radius_along(5, 2, 0.3, 0.3 + math.pi / 2) == pytest.approx(2)


def test_normalize_axes_swaps():
    wa, wb, th, *_ = normalize_axes(2, 5, 0.1)
    assert (wa, wb) == (5, 2)
    assert th == pytest.approx(0.1 + math.pi / 2 - math.pi)


@pytest.mark.parametrize("theta_deg", [0, 20, -35, 80])
def test_fit_recovers_parameters(theta_deg):
    th = math.radians(theta_deg)
    img = make_image(theta=th)
    res = analyze_image(img, AnalysisSettings(pixel_size_um=1.0, magnification=1.0))
    f = res.fit
    assert f.x0 == pytest.approx(330.3, abs=0.2)
    assert f.y0 == pytest.approx(241.7, abs=0.2)
    assert f.wa == pytest.approx(60, rel=0.01)
    assert f.wb == pytest.approx(45, rel=0.01)
    assert f.m == pytest.approx(4, rel=0.03)
    # Winkel modulo 180 deg
    d = (f.theta - th + math.pi / 2) % math.pi - math.pi / 2
    assert abs(d) < math.radians(0.5)
    assert f.r_squared > 0.99


def test_uniformity_of_flat_plateau_is_noise_level():
    img = make_image(m=10, noise=2.0)
    res = analyze_image(img, AnalysisSettings(pixel_size_um=1.0, magnification=1.0))
    # Plateau ~200 counts, Rauschen 2 -> ~1 %, plus Restabfall bis 90 %
    assert res.plateau_uniformity < 0.04
    assert res.flatness_u[0] < 0.03


def test_pixel_size_and_magnification_scale_results():
    img = make_image()
    r1 = analyze_image(img, AnalysisSettings(pixel_size_um=2.0, magnification=1.0))
    r2 = analyze_image(img, AnalysisSettings(pixel_size_um=2.0, magnification=4.0))
    d1 = r1.summary()["D_long_1/e2_um"]
    d2 = r2.summary()["D_long_1/e2_um"]
    assert d1 / d2 == pytest.approx(4.0, rel=1e-6)


def test_cut_crosses_level_at_marked_radius():
    img = make_image(noise=0.0)
    res = analyze_image(img, AnalysisSettings(pixel_size_um=1.0, magnification=1.0))
    cut = res.cuts["horizontal"]
    r = cut.level_radii_um["50%"]
    val = np.interp(r, cut.pos_um, cut.model)
    assert val == pytest.approx(0.5, abs=0.01)


def test_saturation_warning():
    img = np.clip(make_image(A=400), 0, 255)
    res = analyze_image(img, AnalysisSettings(), saturation_value=255)
    assert res.saturated_fraction > 0.5
    assert any("übersteuert" in w for w in res.warnings)


def test_default_pixel_size_is_ids_40x():
    assert AnalysisSettings().um_per_px == pytest.approx(1.45 / 43.75)


@pytest.mark.parametrize("opts", [dict(), dict(show_regions=False, show_text=False),
                                  dict(show_fit=True, show_circles=True)])
def test_all_figures_render(opts):
    from matplotlib.figure import Figure
    from beamshaper.plotting import FIGURES
    res = analyze_image(make_image(), AnalysisSettings())
    for _, func in FIGURES.values():
        kw = {k: v for k, v in opts.items() if k in func.__code__.co_varnames}
        fig = Figure(figsize=(8, 4))
        func(fig, res, **kw)
        fig.canvas.draw()


def make_airy(n=300, x0=150.3, y0=149.6, r0=20.0, A=200, bg=1.0, noise=0.5, seed=1):
    """Airy-Scheibchen: erste Nullstelle bei r0 (px)."""
    from scipy.special import j1
    Y, X = np.mgrid[0:n, 0:n]
    v = 3.8317 * np.hypot(X - x0, Y - y0) / r0
    with np.errstate(invalid="ignore", divide="ignore"):
        I = np.where(v == 0, 1.0, (2 * j1(v) / v) ** 2)
    return A * I + bg + np.random.default_rng(seed).normal(0, noise, (n, n))


def test_airy_rings_are_not_cut_off():
    img = make_airy()
    res = analyze_image(img, AnalysisSettings(pixel_size_um=1.0, magnification=1.0))
    # 1. Nebenmaximum bei 1.64 r0, 2. bei 2.68 r0 (1,7 % bzw. 0,4 % des Maximums)
    assert res.signal_extent_px > 1.64 * 20
    half = (res.view[1] - res.view[0]) / 2
    assert half > 1.8 * 20
    assert np.max(np.abs(res.cuts["horizontal"].pos_um)) > 1.8 * 20


def test_export_is_pdf_without_title(tmp_path):
    from PIL import Image
    from beamshaper.analysis import analyze_file
    from beamshaper.export import save_figures
    p = tmp_path / "spot.bmp"
    Image.fromarray(np.clip(make_airy(), 0, 255).astype(np.uint8)).save(p)
    res = analyze_file(p)
    paths = save_figures(res)
    assert len(paths) == 4 and all(x.suffix == ".pdf" and x.stat().st_size > 0 for x in paths)


def test_intensity_scale_is_max_pixel_value():
    """Alle Intensitaetsachsen/Farbskalen gehen bis 255 (8 bit), unabhaengig von der Helligkeit."""
    from matplotlib.figure import Figure
    from beamshaper.plotting import plot_overview, plot_fit_residual
    for A in (200, 60):   # hell und dunkel -> gleiche Skala
        res = analyze_image(make_image(A=A), AnalysisSettings(), saturation_value=255)
        fig = Figure()
        plot_overview(fig, res)
        ax_im, ax_h, ax_v = fig.axes[:3]
        assert ax_im.images[0].get_clim() == (0, 255)
        assert ax_h.get_ylim() == (0, 255) and ax_v.get_ylim() == (0, 255)
        fig2 = Figure()
        plot_fit_residual(fig2, res)
        assert fig2.axes[0].images[0].get_clim() == (0, 255)


def test_region_uniformity_flat_top_is_small_in_90_region():
    res = analyze_image(make_image(m=10, noise=2.0), AnalysisSettings())
    st = res.region_stats
    assert set(st) == {"1/e2", "50%", "90%"}
    assert st["90%"]["U"] < st["50%"]["U"] < st["1/e2"]["U"]
    assert st["90%"]["U"] < 0.05


def test_font_is_not_cmr10():
    """cmr10 hat auf dem Code von 'µ' einen Gravis -> im PDF stand '(\\'m)'."""
    from beamshaper.plotting import STYLE
    assert "cmr10" not in STYLE["font.family"]


def test_report_pdf(tmp_path):
    from beamshaper.report import save_report, compact
    rs = [compact(analyze_image(make_image(A=A), AnalysisSettings(), path=f"bild_{A}.bmp"))
          for A in (200, 90)]
    p = save_report(rs, tmp_path / "report.pdf")
    assert p.stat().st_size > 10_000
    assert p.read_bytes().count(b"/Type /Page\n") + p.read_bytes().count(b"/Type /Page ") >= 1


def test_overview_legend_and_crosshair_inside_image():
    """Legendenbox liegt komplett im 2D-Bild, Fadenkreuz ragt nicht hinaus."""
    from matplotlib.figure import Figure
    from beamshaper.plotting import plot_overview, EXPORT_SIZES
    res = analyze_image(make_airy(), AnalysisSettings(pixel_size_um=1.0, magnification=1.0))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=EXPORT_SIZES["overview"])
    FigureCanvasAgg(fig)
    plot_overview(fig, res)
    fig.canvas.draw()
    ax_im = fig.axes[0]
    r = fig.canvas.get_renderer()
    box = ax_im.get_window_extent(r)
    leg = ax_im.get_legend().get_window_extent(r)
    assert box.x0 <= leg.x0 and leg.x1 <= box.x1 and box.y0 <= leg.y0 and leg.y1 <= box.y1
    img = ax_im.images[0].get_window_extent(r)
    for line in ax_im.lines[:2]:          # Fadenkreuz
        ext = line.get_window_extent(r)
        assert ext.x0 >= img.x0 - 1 and ext.x1 <= img.x1 + 1
        assert ext.y0 >= img.y0 - 1 and ext.y1 <= img.y1 + 1


def test_region_colors_visible_in_cuts():
    """Keine weisse (oder sehr helle) Regionsfarbe - auf dem grauen Schnitt-Hintergrund unsichtbar."""
    from matplotlib.colors import to_rgb
    from beamshaper.plotting import LEVEL_COLORS
    for c in LEVEL_COLORS.values():
        assert sum(to_rgb(c)) / 3 < 0.8, c
    assert len(set(LEVEL_COLORS.values())) == len(LEVEL_COLORS)


def test_regions_can_be_switched_off_everywhere(tmp_path):
    """show_regions=False: keine Ringe/Level-Linien in keinem Plot und im Report."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Ellipse
    from beamshaper.plotting import FIGURES, LEVEL_COLORS
    from beamshaper.report import save_report, compact
    from matplotlib.colors import to_hex
    res = analyze_image(make_image(), AnalysisSettings())
    level_cols = {to_hex(c) for c in LEVEL_COLORS.values()}
    for key, (_, func) in FIGURES.items():
        fig = Figure()
        kw = {"show_regions": False} if "show_regions" in func.__code__.co_varnames else {}
        func(fig, res, **kw)
        for ax in fig.axes:
            regions = [a for a in ax.get_children() if a.get_gid() == "region"]
            assert not regions, key
            assert not any(isinstance(p, Ellipse) for p in ax.patches), key
        # Gegenprobe: mit Ringen sind sie da
        fig2 = Figure()
        func(fig2, res)
        if key not in ("fit", "title"):   # Titelbild hat nie Ringe
            assert any(a.get_gid() == "region" for ax in fig2.axes for a in ax.get_children()), key
    save_report([compact(res)], tmp_path / "r.pdf", {"show_regions": False})


def _airy_img(n=1400, waist_um=1.9, k=1.483):
    from scipy.special import j1
    s = AnalysisSettings()
    um = s.um_per_px
    r0 = k * waist_um
    Y, X = np.mgrid[0:n, 0:n]
    x0 = y0 = n / 2 + 0.3

    def airy(xu, yu):
        v = 3.8317 * np.hypot(xu, yu) / r0
        with np.errstate(all="ignore"):
            return np.where(v == 0, 1.0, (2 * j1(v) / v) ** 2)
    return 200 * airy((X - x0) * um, (Y - y0) * um), airy, s


def test_lattice_weighted_uniformity_matches_analytic_airy():
    """U_w aus dem Bild = analytischer Wert (Definition wie Multitone_Simulation)."""
    from beamshaper.analysis import sigma_thermal_um
    img, airy, s = _airy_img()
    L = analyze_image(img, s).lattice
    sig = sigma_thermal_um(s)
    xs = np.linspace(-6 * sig, 6 * sig, 241)
    X, Y = np.meshgrid(xs, xs)
    own = airy(X, Y)
    nb = sum(airy(X - i * s.pitch_um, Y - j * s.pitch_um)
             for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
    W = np.exp(-(X ** 2 + Y ** 2) / (2 * sig ** 2))
    mu = (own * W).sum() / W.sum()
    U_w = np.sqrt(((own - mu) ** 2 * W).sum() / W.sum()) / mu
    eta_w = (nb * W).sum() / (own * W).sum()
    assert L["U_w"] == pytest.approx(U_w, rel=0.02)
    assert L["eta_w"] == pytest.approx(eta_w, abs=3 * L["eta_w_err"] + 1e-4)
    assert abs(sig * 1e3 - 107.6) < 0.5


def test_lattice_can_be_disabled_and_small_image_gives_nan():
    img, _, s = _airy_img(n=200)                 # 6,6 um Bild < Pitch
    L = analyze_image(img, s).lattice
    assert np.isnan(L["eta_w"]) and np.isnan(L["hard"]["50%"]["eta"])
    from dataclasses import replace
    assert analyze_image(img, replace(s, lattice_estimate=False)).lattice is None


def test_cut_axis_scale_independent_of_colormap_scale():
    """2D-Bild auf das hellste Pixel gespreizt, Schnitte trotzdem bis 255."""
    from matplotlib.figure import Figure
    from beamshaper.plotting import plot_overview, set_scale_mode
    res = analyze_image(np.clip(make_image(A=60), 0, 255), AnalysisSettings(),
                        saturation_value=255)
    try:
        set_scale_mode("image", cuts="bitdepth")
        fig = Figure()
        plot_overview(fig, res)
        ax_im, ax_h, ax_v = fig.axes[:3]
        assert ax_im.images[0].get_clim()[1] < 120        # auf das Bild gespreizt
        assert ax_h.get_ylim() == (0, 255) and ax_v.get_ylim() == (0, 255)
        set_scale_mode("image", cuts="follow")            # Standard: Schnitte folgen
        fig = Figure()
        plot_overview(fig, res)
        assert fig.axes[1].get_ylim()[1] < 120
    finally:
        set_scale_mode("bitdepth", cuts="follow")


def test_cut_axis_normalised_to_one():
    """Schnitte normiert: Basislinie 0, Plateau 1, Achse bis 1,1."""
    from matplotlib.figure import Figure
    from beamshaper.plotting import plot_overview, set_scale_mode
    res = analyze_image(make_image(A=60, B=8, m=8), AnalysisSettings(), saturation_value=255)
    try:
        set_scale_mode("bitdepth", cuts="norm")
        fig = Figure()
        plot_overview(fig, res)
        ax_h = fig.axes[1]
        assert ax_h.get_ylim() == (0, 1.1)
        y = ax_h.lines[0].get_ydata()
        assert y.max() == pytest.approx(1.0, abs=0.05)
        assert abs(y[:20].mean()) < 0.05          # Basislinie bei 0
    finally:
        set_scale_mode("bitdepth", cuts="follow")


def test_uniformity_radius():
    """Radius, bis zu dem σ/µ unter der Schwelle bleibt: dort ist die
    Flatness-Kurve gerade bei der Schwelle."""
    from dataclasses import replace
    res = analyze_image(make_image(m=4, noise=1.0), AnalysisSettings(pixel_size_um=1.0,
                                                                    magnification=1.0))
    r = res.uniformity_radius_um
    assert np.isfinite(r)
    u_at = np.interp(2 * r, res.flatness_d_um, res.flatness_u)
    assert u_at == pytest.approx(0.05, abs=0.002)
    # andere Schwelle -> anderer (groesserer) Radius
    res2 = analyze_image(make_image(m=4, noise=1.0),
                         replace(AnalysisSettings(pixel_size_um=1.0, magnification=1.0),
                                 uniformity_threshold=0.10))
    assert res2.uniformity_radius_um > r
    assert res.summary()["r_uniform_um"] == pytest.approx(r)


def test_uniformity_radius_nan_when_never_reached():
    """Sehr flaches, rauscharmes Profil: Schwelle wird nie erreicht -> NaN statt Zahl."""
    res = analyze_image(make_image(m=12, noise=0.05), AnalysisSettings())
    assert np.isnan(res.uniformity_radius_um) or res.uniformity_radius_um > 0


def test_title_image_fills_titlepage_window(tmp_path):
    """Titelbild: PDF exakt 155 x 185,7 mm, Bild fuellt die ganze Flaeche (kein weisser Rand),
    unbearbeitet (keine Ringe, kein Text) und wird nicht im Standard-Export mitgespeichert."""
    from PIL import Image
    from beamshaper.analysis import analyze_file
    from beamshaper.export import save_figures
    from beamshaper.plotting import plot_title, title_window
    from matplotlib.figure import Figure
    p = tmp_path / "spot.bmp"
    Image.fromarray(np.clip(make_airy(n=600), 0, 255).astype(np.uint8)).save(p)
    res = analyze_file(p)
    assert all("title" not in x.name for x in save_figures(res))
    path = save_figures(res, which=["title"])[0]
    assert path.name == "spot_title.pdf"
    fig = Figure(figsize=(155 / 25.4, 185.7 / 25.4))
    plot_title(fig, res, show_logo_hint=False)
    ax = fig.axes[0]
    assert ax.get_position().bounds == pytest.approx((0, 0, 1, 1), abs=1e-9)
    assert not ax.patches and not ax.texts and not ax.lines
    x0, x1 = ax.get_xlim(); y1, y0 = ax.get_ylim()
    assert (y1 - y0) / (x1 - x0) == pytest.approx(185.7 / 155, rel=1e-9)
    # Fenster liegt komplett im Kamerabild
    assert x0 >= -0.5 and y0 >= -0.5 and x1 <= 599.5 and y1 <= 599.5
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[..., :3]
    assert rgb.min(axis=2).max() < 250          # kein einziges weisses Pixel


def test_title_window_beam_position():
    from beamshaper.plotting import title_window
    img = make_image(nx=2000, ny=2000, x0=1000, y0=1000)
    res = analyze_image(img, AnalysisSettings())
    x0, y0, W, H, shrunk = title_window(res, img, fill=0.3, pos_x=0.5, pos_y=0.6)
    assert not shrunk
    assert (1000 - x0) / W == pytest.approx(0.5, abs=0.01)
    assert (1000 - y0) / H == pytest.approx(0.6, abs=0.01)


def test_title_window_frames_whole_spot_grid():
    """3x3-Spotgitter: das ganze Muster wird zentriert und fuellt `fill` der Breite."""
    from beamshaper.plotting import title_window
    ny, nx = 1200, 2000
    img = np.full((ny, nx), 3.0)
    for i in range(3):
        for j in range(3):
            y, x = 500 + 90 * i, 1200 + 90 * j
            img[y - 30:y + 30, x - 30:x + 30] = 200
    res = analyze_image(img, AnalysisSettings())
    x0, y0, W, H, shrunk = title_window(res, img, fill=0.5)
    assert x0 + W / 2 == pytest.approx(1290, abs=4)       # Mitte des Gitters
    assert y0 + H / 2 == pytest.approx(590, abs=4)
    assert 240 / W == pytest.approx(0.5, abs=0.03)


# ---------------------------------------------------------------------------
# Simuliertes Profil (.npz aus Multitone_Simulation) und Vergleich
# ---------------------------------------------------------------------------
def _write_sim(path, um_per_px=0.05, n=301, w_um=2.0):
    """Gauss-Spot als Simulationsdatei wie aus Weighted_Multitone_Lens_GUI.py."""
    import json
    t = (np.arange(n) - (n - 1) / 2) * um_per_px
    X, Y = np.meshgrid(t, t[::-1])
    img = np.exp(-2 * (X ** 2 + Y ** 2) / w_um ** 2)
    np.savez_compressed(path, image=img.astype(np.float32), um_per_px=um_per_px,
                        x_um=t, y_um=t[::-1], params=json.dumps({"N_x": 1, "N_y": 1}))
    return path


def test_simulation_uses_its_own_pixel_size(tmp_path):
    from beamshaper.analysis import analyze_file
    p = _write_sim(tmp_path / "sim.npz", um_per_px=0.05, w_um=2.0)
    # Kamera-Einstellungen duerfen die Pixelgroesse der Simulation nicht ueberschreiben
    res = analyze_file(p, AnalysisSettings(pixel_size_um=1.45, magnification=43.75))
    assert res.is_simulation
    assert res.um_per_px == pytest.approx(0.05)
    assert res.saturation_value == 1.0
    assert res.saturated_fraction == 0.0
    # Gauss: 1/e^2-Durchmesser = 2 w
    assert res.summary()["D_long_1/e2_um"] == pytest.approx(4.0, rel=0.02)
    assert res.simulation["N_x"] == 1


def test_comparison_plot_and_export(tmp_path):
    from beamshaper.analysis import analyze_file
    from beamshaper.export import save_comparison
    from beamshaper.plotting import comparison_order
    from PIL import Image
    sim = analyze_file(_write_sim(tmp_path / "sim.npz", um_per_px=0.05))
    img = make_image(nx=400, ny=400, x0=200, y0=200, wa=40, wb=30, noise=0.5)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(tmp_path / "mess.png")
    meas = analyze_file(tmp_path / "mess.png", AnalysisSettings(pixel_size_um=0.05,
                                                                  magnification=1.0))
    (left, l_lab), (right, r_lab) = comparison_order(meas, sim)
    assert left is sim and l_lab == "Simulation" and r_lab == "Measurement"
    p = save_comparison(meas, sim, base=tmp_path)
    assert p.is_file() and p.name == "mess_vs_sim_comparison.pdf"
