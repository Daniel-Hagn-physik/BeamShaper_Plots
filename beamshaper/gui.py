"""PyQt5-Oberflaeche fuer die FlatTop-Auswertung.

Start:  python main.py            (oder: python main.py <bild|ordner>)
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rcParams["savefig.format"] = "pdf"   # Speichern-Knopf der Toolbar -> PDF
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QThread, QTimer, QSettings, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QListWidget, QListWidgetItem, QDoubleSpinBox, QSpinBox,
    QComboBox, QCheckBox, QLineEdit, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QProgressBar, QLabel, QSplitter,
    QScrollArea, QToolButton, QDialog, QDialogButtonBox,
)

from .analysis import AnalysisSettings, FlatTopResult, analyze_file
from .batch import run_batch
from .export import save_comparison, save_figures, save_report_for, save_results
from .imageio import IMAGE_SUFFIXES, is_background_file, is_simulation_file, list_images
from .plotting import FIGURES, plot_comparison, set_font_size, set_scale_mode

APP_NAME = "BeamShaperPlots"
LOG_FILE = None   # wird von main.py gesetzt (crash.log)


def log(msg: str) -> None:
    """In Konsole und crash.log schreiben."""
    print(msg, file=sys.stderr, flush=True)
    if LOG_FILE is not None:
        try:
            LOG_FILE.write(msg + "\n")
            LOG_FILE.flush()
        except Exception:
            pass
# Standard-Datenordner im Repo; hier liegen die Messbilder
DATA_DIR = Path(__file__).resolve().parents[1] / "Messdaten"
# Dateifilter fuer alle Oeffnen-Dialoge: Kamerabilder und simulierte Profile (.npz)
FILE_FILTER = ("Bilder und Simulationen (" + " ".join("*" + s for s in IMAGE_SUFFIXES) + ")")


# ---------------------------------------------------------------------------
# Hintergrund-Threads
# ---------------------------------------------------------------------------
class AnalysisThread(QThread):
    done = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, req_id: int, path: str, settings: AnalysisSettings):
        super().__init__()
        self.req_id, self.path, self.settings = req_id, path, settings

    def run(self):
        try:
            self.done.emit(self.req_id, analyze_file(self.path, self.settings))
        except Exception as exc:
            self.failed.emit(self.req_id, f"{exc}\n\n{traceback.format_exc()}")


class BatchThread(QThread):
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(str, int)
    failed = pyqtSignal(str)

    def __init__(self, folder, settings, save_pdf, files, plot_opts=None, lattice_files=None):
        super().__init__()
        self.plot_opts = plot_opts or {}
        self.lattice_files = lattice_files
        self.folder, self.settings, self.save_pdf, self.files = folder, settings, save_pdf, files
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        def cb(i, n, name):
            self.progress.emit(i, n, name)
            return not self._cancel
        try:
            out, rows = run_batch(self.folder, self.settings, save_pdf=self.save_pdf,
                                  progress=cb, files=self.files, plot_opts=self.plot_opts,
                                  lattice_files=self.lattice_files)
            self.done.emit(str(out), len(rows))
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Hauptfenster
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BeamShaperPlots - FlatTop-Auswertung")
        self.resize(1400, 850)
        self.qs = QSettings(APP_NAME, APP_NAME + "_v2")  # v2: neue Standard-Pixelgroesse

        self.folder: Path | None = None
        self.current_path: str | None = None
        self.result: FlatTopResult | None = None
        self.cmp_result: FlatTopResult | None = None    # Vergleichsbild (z. B. Simulation)
        self._req = 0
        self._req_cmp = 0
        self._threads: list[QThread] = []
        self.batch_thread: BatchThread | None = None

        self._debounce = QTimer(self, singleShot=True, interval=400)
        self._debounce.timeout.connect(self.run_analysis)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 1010])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Bild oder Ordner öffnen")
        self._load_settings()
        self._update_sigma()

    # -- Aufbau -------------------------------------------------------------
    def _build_left(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        row = QHBoxLayout()
        b_file = QPushButton("Bild öffnen...")
        b_file.clicked.connect(self.open_file)
        b_dir = QPushButton("Ordner öffnen...")
        b_dir.clicked.connect(self.open_folder)
        row.addWidget(b_file)
        row.addWidget(b_dir)
        lay.addLayout(row)

        self.lbl_folder = QLabel("-")
        self.lbl_folder.setWordWrap(True)
        self.lbl_folder.setStyleSheet("color: #52514e;")
        lay.addWidget(self.lbl_folder)

        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self._on_file_selected)
        lay.addWidget(self.file_list, stretch=1)

        # Parameter
        g = QGroupBox("Parameter")
        f = QFormLayout(g)
        self.sp_pixel = self._dspin(0.001, 100, 3, 1.45, " µm")
        self.sp_mag = self._dspin(0.001, 1000, 3, 43.75, " x")
        self.sp_roi = self._dspin(1.2, 8, 2, 2.5)
        self.sp_fitpx = QSpinBox(); self.sp_fitpx.setRange(50, 2000); self.sp_fitpx.setValue(250)
        self.sp_plevel = self._dspin(0.5, 0.99, 2, 0.9)
        self.sp_pscale = self._dspin(0.1, 1.0, 2, 1.0)
        self.sp_fstep = QSpinBox(); self.sp_fstep.setRange(1, 100); self.sp_fstep.setValue(2)
        self.sp_cut = self._dspin(0, 1e5, 1, 0, " µm")
        self.sp_cut.setSpecialValueText("auto")
        f.addRow("Pixelgröße (Kamera)", self.sp_pixel)
        f.addRow("Vergrößerung", self.sp_mag)
        self.lbl_umpx = QLabel()
        self.lbl_umpx.setStyleSheet("color: #52514e;")
        f.addRow("= µm/Pixel (Objekt)", self.lbl_umpx)
        for sp in (self.sp_pixel, self.sp_mag):
            sp.valueChanged.connect(self._update_umpx)
        self._update_umpx()
        self.ed_bg = QLineEdit(); self.ed_bg.setPlaceholderText("kein Hintergrundbild")
        self.ed_bg.setReadOnly(True)
        bgrow = QHBoxLayout()
        bgrow.addWidget(self.ed_bg)
        b_bg = QToolButton(text="..."); b_bg.clicked.connect(self.choose_background)
        b_bgx = QToolButton(text="x"); b_bgx.clicked.connect(lambda: self._set_background(""))
        bgrow.addWidget(b_bg); bgrow.addWidget(b_bgx)
        f.addRow("Hintergrund", bgrow)
        f.addRow("ROI-Faktor (x r50)", self.sp_roi)
        f.addRow("Fit: max. Pixel", self.sp_fitpx)
        f.addRow("Plateau-Level", self.sp_plevel)
        f.addRow("Plateau-Skala", self.sp_pscale)
        f.addRow("Flatness-Schritt (px)", self.sp_fstep)
        self.sp_uthr = self._dspin(0.1, 50, 1, 5.0, " %")
        self.sp_uthr.setToolTip("Schwelle für den Uniformity-Radius: größter Kreis,\n"
                                "in dem σ/µ noch darunter liegt (steht im Report).")
        f.addRow("Uniformity-Schwelle", self.sp_uthr)
        f.addRow("Schnitt-Halbbreite", self.sp_cut)
        for sp in (self.sp_pixel, self.sp_mag, self.sp_roi, self.sp_fitpx, self.sp_plevel,
                   self.sp_pscale, self.sp_fstep, self.sp_cut, self.sp_uthr):
            sp.valueChanged.connect(lambda *_: self._debounce.start())
        lay.addWidget(g)

        # Abschaetzung im Fallengitter (Multitone_Simulation-Definitionen)
        g3 = QGroupBox("Uniformity/Crosstalk im Fallengitter")
        f3 = QFormLayout(g3)
        self.chk_lattice = QCheckBox("Abschätzung berechnen")
        self.chk_lattice.setChecked(True)
        self.chk_lattice.setToolTip("Hart (Ringe 1/e², 50 %, 90 %) und atomgewichtet,\n"
                                    "wie in Multitone_Simulation. Für große Profile abschalten.")
        self.sp_pitch = self._dspin(0.1, 1000, 3, 5.288, " µm")
        self.sp_temp = self._dspin(0.1, 1000, 1, 17.0, " µK")
        self.sp_nu = self._dspin(0.1, 1000, 1, 60.4, " kHz")
        f3.addRow(self.chk_lattice)
        f3.addRow("Pitch", self.sp_pitch)
        f3.addRow("Temperatur T", self.sp_temp)
        f3.addRow("Fallenfrequenz ν_r", self.sp_nu)
        self.lbl_sigma = QLabel()
        self.lbl_sigma.setStyleSheet("color: #52514e;")
        f3.addRow("σ_atom (Rb-85)", self.lbl_sigma)
        for sp in (self.sp_pitch, self.sp_temp, self.sp_nu):
            sp.valueChanged.connect(lambda *_: self._debounce.start())
            sp.valueChanged.connect(self._update_sigma)
        self.chk_lattice.toggled.connect(lambda *_: self._debounce.start())
        lay.addWidget(g3)

        # Darstellung
        g2 = QGroupBox("Darstellung / Export")
        f2 = QFormLayout(g2)
        self.cb_cmap = QComboBox(); self.cb_cmap.addItems(["jet", "inferno", "viridis", "gray"])
        self.chk_ell = QCheckBox("Ringe 1/e², 50 %, 90 % (alle Plots + Report)"); self.chk_ell.setChecked(True)
        self.chk_text = QCheckBox("Weißer Text im 2D-Plot (FWHM)"); self.chk_text.setChecked(True)
        self.chk_fit = QCheckBox("Fit in den Schnitten")
        self.chk_circ = QCheckBox("Flatness-Kreise")
        for wdg in (self.cb_cmap,):
            wdg.currentIndexChanged.connect(self.redraw)
        for chk in (self.chk_ell, self.chk_text, self.chk_fit, self.chk_circ):
            chk.toggled.connect(self.redraw)
        f2.addRow("Farbskala", self.cb_cmap)
        self.cb_scale = QComboBox()
        self.cb_scale.addItems(["0 … max. Pixelwert (255)", "0 … hellster Pixel im Bild"])
        self.cb_scale.setToolTip("Standard: feste Skala bis zum maximalen Pixelwert der "
                                 "Bittiefe,\ndamit alle Bilder gleich normiert und "
                                 "vergleichbar sind.")
        self.cb_scale.currentIndexChanged.connect(self.redraw)
        f2.addRow("Farbskala 2D-Bild", self.cb_scale)
        self.cb_cutscale = QComboBox()
        self.cb_cutscale.addItems(["wie Farbskala", "0 … max. Pixelwert (255)",
                                   "0 … hellster Pixel im Bild", "normiert auf 1"])
        self.cb_cutscale.setToolTip("y-Achse der Schnitte. Beispiel: 2D-Bild auf den "
                                    "hellsten Pixel spreizen (dunkle Aufnahmen sichtbar),\n"
                                    "Schnitte trotzdem bis 255 - dann bleiben sie zwischen "
                                    "Bildern vergleichbar.")
        self.cb_cutscale.currentIndexChanged.connect(self.redraw)
        f2.addRow("y-Achse Schnitte", self.cb_cutscale)
        self.sp_font = self._dspin(6, 20, 1, 11, " pt")
        self.sp_font.setToolTip("Eine Schriftgröße für alle Texte. Plots sind 16 cm breit:\n"
                                "mit width=\\textwidth erscheint sie in LaTeX genau so groß.")
        self.sp_font.valueChanged.connect(self.redraw)
        f2.addRow("Schriftgröße (LaTeX)", self.sp_font)
        f2.addRow(self.chk_ell)
        f2.addRow(self.chk_text)
        f2.addRow(self.chk_fit)
        f2.addRow(self.chk_circ)
        lay.addWidget(g2)

        # Titelbild fuer die Titelseite der Thesis
        g4 = QGroupBox("Titelbild (Titelseite Thesis)")
        f4 = QFormLayout(g4)
        tip = ("Unbearbeitetes Kamerabild im gewählten Farbschema, ohne Rand, Achsen,\n"
               "Fadenkreuz oder Ringe. PDF exakt so groß wie das Bildfenster der\n"
               "Titelseite (zwischen den beiden Strichen).\n"
               "In LaTeX: \\titleimage*{\\includegraphics{<name>_title.pdf}}")
        g4.setToolTip(tip)
        self.sp_tfill = self._dspin(5, 100, 0, 45, " %")
        self.sp_tfill.setToolTip("Strahl-Durchmesser (inkl. Nebenstruktur) als Anteil der Bildbreite.\n"
                                 "Ist das Kamerabild zu klein, wird der Ausschnitt verkleinert.")
        self.sp_tx = self._dspin(0, 100, 0, 50, " %")
        self.sp_tx.setToolTip("Strahlzentrum horizontal im Fenster (0 = links, 100 = rechts)")
        self.sp_ty = self._dspin(0, 100, 0, 50, " %")
        self.sp_ty.setToolTip("Strahlzentrum vertikal im Fenster (0 = oben, 100 = unten).\n"
                              "Die Logos liegen oben rechts.")
        self.sp_tw = self._dspin(10, 300, 1, 155.0, " mm")
        self.sp_th = self._dspin(10, 300, 1, 185.7, " mm")
        self.sp_tw.setToolTip("Breite des Bildfensters = \\textwidth (A4, 31/24 mm Rand)")
        self.sp_th.setToolTip("Höhe vom Strich unter dem Titelblock bis zum Strich unten\n"
                              "(gemessen in der APQ-Vorlage)")
        for sp in (self.sp_tfill, self.sp_tx, self.sp_ty, self.sp_tw, self.sp_th):
            sp.valueChanged.connect(self.redraw)
        self.chk_tbg = QCheckBox("Hintergrund abziehen")
        self.chk_tbg.setToolTip("Zieht das oben unter „Hintergrund“ gewählte Bild ab\n"
                                "(negative Werte werden 0). Ohne Hintergrundbild: unbearbeitet.")
        self.chk_tbg.toggled.connect(self.redraw)
        f4.addRow(self.chk_tbg)
        f4.addRow("Strahlgröße (Anteil Breite)", self.sp_tfill)
        f4.addRow("Position x", self.sp_tx)
        f4.addRow("Position y", self.sp_ty)
        f4.addRow("Breite", self.sp_tw)
        f4.addRow("Höhe", self.sp_th)
        b_title = QPushButton("Titelbild als PDF speichern")
        b_title.setToolTip(tip)
        b_title.clicked.connect(self.export_title)
        f4.addRow(b_title)
        lay.addWidget(g4)

        # Vergleich mit einem zweiten Profil, z. B. Simulation aus Multitone_Simulation
        g5 = QGroupBox("Vergleich (z. B. Simulation)")
        f5 = QFormLayout(g5)
        tip5 = ("Zweites Profil neben dem aktuellen Bild, Reiter „Vergleich“.\n"
                "Simulation: .npz aus Weighted_Multitone_Lens_GUI.py\n"
                "(Knopf „Export 2D profile for BeamShaperPlots“) - bringt ihre\n"
                "Pixelgröße selbst mit. Simulation steht links, beide auf ihr\n"
                "Maximum normiert, gleicher Ausschnitt in µm.")
        g5.setToolTip(tip5)
        self.ed_cmp = QLineEdit(); self.ed_cmp.setPlaceholderText("kein Vergleichsprofil")
        self.ed_cmp.setReadOnly(True)
        self.ed_cmp.setToolTip(tip5)
        crow = QHBoxLayout()
        crow.addWidget(self.ed_cmp)
        b_cmp = QToolButton(text="..."); b_cmp.clicked.connect(self.choose_comparison)
        b_cmpx = QToolButton(text="x"); b_cmpx.clicked.connect(lambda: self._set_comparison(""))
        crow.addWidget(b_cmp); crow.addWidget(b_cmpx)
        f5.addRow("Profil", crow)
        b_cmp_save = QPushButton("Vergleich als PDF speichern")
        b_cmp_save.setToolTip("Auswertung/Plots/JJJJ-MM-TT/<Messung>_vs_<Simulation>_comparison.pdf\n"
                              "(16 cm breit, neben dem Messbild)")
        b_cmp_save.clicked.connect(self.export_comparison)
        f5.addRow(b_cmp_save)
        lay.addWidget(g5)

        b_run = QPushButton("Neu auswerten")
        b_run.clicked.connect(self.run_analysis)
        b_exp = QPushButton("Exportieren (Plots + Report als PDF, JSON, CSV)")
        b_exp.clicked.connect(self.export_current)
        lay.addWidget(b_run)
        lay.addWidget(b_exp)

        gb = QGroupBox("Batch")
        vb = QVBoxLayout(gb)
        self.chk_batch_pdf = QCheckBox("PDF-Plots für jedes Bild")
        self.chk_batch_pdf.setChecked(True)
        self.btn_batch = QPushButton("Ganzen Ordner auswerten")
        self.btn_batch.clicked.connect(self.start_batch)
        self.btn_cancel = QPushButton("Abbrechen")
        self.btn_cancel.clicked.connect(self.cancel_batch)
        self.btn_cancel.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        vb.addWidget(self.chk_batch_pdf)
        hb = QHBoxLayout(); hb.addWidget(self.btn_batch); hb.addWidget(self.btn_cancel)
        vb.addLayout(hb)
        vb.addWidget(self.progress)
        lay.addWidget(gb)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        scroll.setMinimumWidth(370)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    def _build_right(self) -> QWidget:
        self.tabs = QTabWidget()
        self.figs: dict[str, tuple[Figure, FigureCanvas]] = {}
        for key, (title, _) in FIGURES.items():
            page = QWidget()
            v = QVBoxLayout(page)
            fig = Figure(figsize=(8, 5))
            canvas = FigureCanvas(fig)
            v.addWidget(NavigationToolbar(canvas, page))
            v.addWidget(canvas, stretch=1)
            self.figs[key] = (fig, canvas)
            self.tabs.addTab(page, title.replace("&", "&&"))

        # Vergleich zweier Profile (eigene Figur, braucht zwei Ergebnisse)
        page = QWidget()
        v = QVBoxLayout(page)
        self.fig_cmp = Figure(figsize=(8, 4))
        self.canvas_cmp = FigureCanvas(self.fig_cmp)
        v.addWidget(NavigationToolbar(self.canvas_cmp, page))
        v.addWidget(self.canvas_cmp, stretch=1)
        self.tabs.addTab(page, "Vergleich")

        page = QWidget()
        v = QVBoxLayout(page)
        self.lbl_warn = QLabel("")
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet("color: #b3261e; font-weight: bold;")
        v.addWidget(self.lbl_warn)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Größe", "Wert"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.table)
        self.tabs.addTab(page, "Ergebnisse")
        return self.tabs

    def _update_umpx(self, *_):
        self.lbl_umpx.setText(f"{self.sp_pixel.value() / self.sp_mag.value():.5f} µm")

    @staticmethod
    def _dspin(lo, hi, dec, val, suffix=""):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(dec)
        sp.setSingleStep(10 ** -dec if dec <= 2 else 0.01)
        sp.setValue(val)
        sp.setSuffix(suffix)
        sp.setKeyboardTracking(False)
        return sp

    # -- Einstellungen ------------------------------------------------------
    def settings(self) -> AnalysisSettings:
        return replace(
            AnalysisSettings(),
            pixel_size_um=self.sp_pixel.value(),
            magnification=self.sp_mag.value(),
            background_path=self.ed_bg.text() or None,
            roi_factor=self.sp_roi.value(),
            fit_max_pixels=self.sp_fitpx.value(),
            plateau_level=self.sp_plevel.value(),
            plateau_scale=self.sp_pscale.value(),
            flatness_step_px=self.sp_fstep.value(),
            uniformity_threshold=self.sp_uthr.value() / 100,
            cut_halfwidth_um=self.sp_cut.value(),
            lattice_estimate=self.chk_lattice.isChecked(),
            pitch_um=self.sp_pitch.value(),
            atom_temperature_uK=self.sp_temp.value(),
            trap_freq_kHz=self.sp_nu.value(),
        )

    def _update_sigma(self, *_):
        from .analysis import sigma_thermal_um
        sig = sigma_thermal_um(replace(AnalysisSettings(), atom_temperature_uK=self.sp_temp.value(),
                                       trap_freq_kHz=self.sp_nu.value()))
        self.lbl_sigma.setText(f"{sig * 1e3:.0f} nm")

    _PERSIST = ["sp_pixel", "sp_mag", "sp_roi", "sp_fitpx", "sp_plevel", "sp_pscale",
                "sp_fstep", "sp_cut", "sp_font", "sp_uthr", "sp_pitch", "sp_temp", "sp_nu",
                "sp_tfill", "sp_tx", "sp_ty", "sp_tw", "sp_th"]

    _PERSIST_CHK = ["chk_ell", "chk_text", "chk_fit", "chk_circ", "chk_batch_pdf", "chk_lattice",
                    "chk_tbg"]

    def _load_settings(self):
        for name in self._PERSIST:
            v = self.qs.value(name)
            if name == "sp_th" and v is not None and abs(float(v) - 184.9) < 1e-6:
                v = None          # alter Standard (APQ-Vorlage) -> neuer Standard der Thesis
            if v is not None:
                sp = getattr(self, name)
                sp.blockSignals(True)
                sp.setValue(type(sp.value())(float(v)))
                sp.blockSignals(False)
        for name in self._PERSIST_CHK:
            v = self.qs.value(name)
            if v is not None:
                getattr(self, name).setChecked(str(v).lower() in ("true", "1"))
        cm = self.qs.value("cmap")
        if cm:
            self.cb_cmap.setCurrentText(cm)
        for name in ("cb_scale", "cb_cutscale"):
            v = self.qs.value(name)
            if v is not None:
                getattr(self, name).setCurrentIndex(int(v))
        cmp_path = self.qs.value("cmp_path")
        if cmp_path and Path(cmp_path).is_file():
            self.ed_cmp.setText(str(cmp_path))     # ausgewertet wird mit dem ersten Bild

    def closeEvent(self, ev):
        for name in self._PERSIST:
            self.qs.setValue(name, getattr(self, name).value())
        self.qs.setValue("cmap", self.cb_cmap.currentText())
        for name in ("cb_scale", "cb_cutscale"):
            self.qs.setValue(name, getattr(self, name).currentIndex())
        for name in self._PERSIST_CHK:
            self.qs.setValue(name, getattr(self, name).isChecked())
        if self.folder:
            self.qs.setValue("last_dir", str(self.folder))
        self.qs.setValue("cmp_path", self.ed_cmp.text())
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.cancel()
            self.batch_thread.wait(5000)
        for th in self._threads:
            th.wait(5000)
        super().closeEvent(ev)

    # -- Dateien ------------------------------------------------------------
    def _start_dir(self) -> str:
        if self.folder:
            return str(self.folder)
        last = self.qs.value("last_dir")
        if last and Path(last).is_dir():
            return str(last)
        return str(DATA_DIR if DATA_DIR.is_dir() else Path.home())

    def choose_start_file(self):
        """Beim Programmstart: Datei auswaehlen (Start im Ordner Messdaten)."""
        filt = FILE_FILTER
        start = str(DATA_DIR) if DATA_DIR.is_dir() else self._start_dir()
        path, _ = QFileDialog.getOpenFileName(self, "Welche Datei soll ausgewertet werden?",
                                              start, filt)
        if path:
            self.load_path(path)

    def open_file(self):
        filt = FILE_FILTER
        path, _ = QFileDialog.getOpenFileName(self, "Bild öffnen", self._start_dir(), filt)
        if path:
            self.load_path(path)

    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Ordner öffnen", self._start_dir())
        if d:
            self.load_path(d)

    def load_path(self, path: str):
        p = Path(path)
        folder = p if p.is_dir() else p.parent
        self.folder = folder
        self.lbl_folder.setText(str(folder))
        self.file_list.blockSignals(True)
        self.file_list.clear()
        select = None
        for f in list_images(folder, skip_background=False):
            it = QListWidgetItem(f.name)
            it.setData(Qt.UserRole, str(f))
            if is_background_file(f):
                it.setForeground(QColor("#8a8984"))
                it.setToolTip("Hintergrundbild - wird im Batch übersprungen")
            elif is_simulation_file(f):
                it.setForeground(QColor("#1f5fa8"))
                it.setToolTip("Simuliertes Profil (Multitone_Simulation), "
                              "Pixelgröße aus der Datei")
            self.file_list.addItem(it)
            if p.is_file() and f == p:
                select = it
        self.file_list.blockSignals(False)
        if select is None and self.file_list.count():
            for i in range(self.file_list.count()):
                if not is_background_file(self.file_list.item(i).text()):
                    select = self.file_list.item(i)
                    break
        if select is not None:
            self.file_list.setCurrentItem(select)
        self.qs.setValue("last_dir", str(folder))

    def _on_file_selected(self, cur, _prev):
        if cur is None:
            return
        self.current_path = cur.data(Qt.UserRole)
        self.run_analysis()

    def choose_background(self):
        filt = FILE_FILTER
        path, _ = QFileDialog.getOpenFileName(self, "Hintergrundbild", self._start_dir(), filt)
        if path:
            self._set_background(path)

    def _set_background(self, path: str):
        self.ed_bg.setText(path)
        self._debounce.start()

    # -- Vergleichsprofil ---------------------------------------------------
    def choose_comparison(self):
        path, _ = QFileDialog.getOpenFileName(self, "Vergleichsprofil (z. B. Simulation .npz)",
                                              self._start_dir(), FILE_FILTER)
        if path:
            self._set_comparison(path)

    def _set_comparison(self, path: str):
        self.ed_cmp.setText(path)
        self.qs.setValue("cmp_path", path)
        if not path:
            self.cmp_result = None
            self._req_cmp += 1
            self.redraw_comparison()
            return
        self.run_comparison()
        self.tabs.setCurrentIndex(self.tabs.count() - 2)     # Reiter "Vergleich"

    def run_comparison(self):
        path = self.ed_cmp.text()
        if not path:
            return
        self._req_cmp += 1
        self._threads = [th for th in self._threads if not th.isFinished()]
        t = AnalysisThread(self._req_cmp, path, self.settings())
        t.done.connect(self._on_cmp_result)
        t.failed.connect(self._on_cmp_failed)
        self._threads.append(t)
        t.start()

    def _on_cmp_result(self, req_id: int, res: FlatTopResult):
        if req_id != self._req_cmp:
            return
        self.cmp_result = res
        self.redraw_comparison()

    def _on_cmp_failed(self, req_id: int, msg: str):
        if req_id != self._req_cmp:
            return
        log("Vergleichsprofil: " + msg)
        QMessageBox.warning(self, "Vergleichsprofil", msg.split("\n\n")[0])

    def redraw_comparison(self, *_):
        fig = self.fig_cmp
        try:
            if self.result is None or self.cmp_result is None:
                fig.clear()
                fig.text(0.5, 0.5, "Bild links auswählen und unter „Vergleich“ ein zweites\n"
                         "Profil wählen (z. B. Simulation .npz aus Multitone_Simulation).",
                         ha="center", va="center", color="#52514e")
            else:
                self._plot_opts()            # Schriftgroesse setzen
                plot_comparison(fig, self.result, self.cmp_result,
                                cmap=self.cb_cmap.currentText())
        except Exception as exc:
            log("Plot 'Vergleich': " + traceback.format_exc())
            fig.clear()
            fig.text(0.5, 0.5, f"Plot fehlgeschlagen:\n{exc}", ha="center", va="center",
                     color="#b3261e", wrap=True)
        self.canvas_cmp.draw_idle()

    def export_comparison(self):
        if self.result is None or self.cmp_result is None:
            QMessageBox.information(self, "Vergleich",
                                    "Es braucht ein ausgewertetes Bild und ein Vergleichsprofil.")
            return
        try:
            p = save_comparison(self.result, self.cmp_result, **self._plot_opts())
        except Exception as exc:
            log("Vergleich: " + traceback.format_exc())
            QMessageBox.warning(self, "Vergleich", str(exc))
            return
        self.statusBar().showMessage(f"Vergleich gespeichert: {p}")

    # -- Auswertung ---------------------------------------------------------
    def run_analysis(self):
        if not self.current_path:
            return
        self._req += 1
        # Referenzen auf Threads behalten, bis sie WIRKLICH beendet sind - ein
        # QThread, der geloescht wird waehrend er noch laeuft, beendet sonst den
        # ganzen Prozess (Windows: 0xC0000409).
        self._threads = [th for th in self._threads if not th.isFinished()]
        t = AnalysisThread(self._req, self.current_path, self.settings())
        t.done.connect(self._on_result)
        t.failed.connect(self._on_failed)
        self._threads.append(t)
        self.statusBar().showMessage(f"Werte {Path(self.current_path).name} aus ...")
        t.start()
        # Vergleichsprofil mit denselben Einstellungen neu auswerten
        if self.ed_cmp.text() and (self.cmp_result is None
                                   or self.cmp_result.settings != self._cmp_settings()):
            self.run_comparison()

    def _cmp_settings(self):
        """Einstellungen, mit denen das Vergleichsprofil ausgewertet wuerde
        (Simulation: Pixelgroesse aus der Datei, kein Hintergrund)."""
        st = self.settings()
        res = self.cmp_result
        if res is not None and res.is_simulation:
            st = replace(st, pixel_size_um=res.settings.pixel_size_um, magnification=1.0,
                         background_path=None)
        return st

    def _on_result(self, req_id: int, res: FlatTopResult):
        if req_id != self._req:
            return  # veraltet (inzwischen anderes Bild/andere Parameter)
        self.result = res
        self.redraw()
        try:
            self._fill_table(res)
        except Exception as exc:
            log("Ergebnistabelle: " + traceback.format_exc())
            self.statusBar().showMessage(f"Ergebnistabelle fehlgeschlagen: {exc}")
            return
        f = res.fit
        self.statusBar().showMessage(
            f"{res.name}: m = {f.m:.2f}, R² = {f.r_squared:.4f}, Binning {f.binning}x")

    def _on_failed(self, req_id: int, msg: str):
        if req_id != self._req:
            return
        self.statusBar().showMessage("Auswertung fehlgeschlagen")
        log("Auswertung: " + msg)
        QMessageBox.warning(self, "Fehler", msg.split("\n\n")[0])

    def redraw(self, *_):
        res = self.result
        if res is None:
            return
        opts = self._plot_opts()
        errors = []
        for key, (title, func) in FIGURES.items():
            fig, canvas = self.figs[key]
            kw = {k: v for k, v in opts.items() if k in func.__code__.co_varnames}
            try:
                func(fig, res, **kw)
            except Exception as exc:   # ein Plot darf nie die ganze GUI beenden
                errors.append(f"{title}: {exc}")
                log(f"Plot '{title}': " + traceback.format_exc())
                fig.clear()
                fig.text(0.5, 0.5, f"Plot fehlgeschlagen:\n{exc}", ha="center", va="center",
                         color="#b3261e", wrap=True)
            canvas.draw_idle()
        self.redraw_comparison()
        if errors:
            self.statusBar().showMessage("Plotfehler: " + " | ".join(errors))

    def _plot_opts(self) -> dict:
        set_font_size(self.sp_font.value())
        set_scale_mode("image" if self.cb_scale.currentIndex() == 1 else "bitdepth",
                       cuts=("follow", "bitdepth", "image",
                             "norm")[self.cb_cutscale.currentIndex()])
        return dict(cmap=self.cb_cmap.currentText(), show_regions=self.chk_ell.isChecked(),
                    show_text=self.chk_text.isChecked(), show_fit=self.chk_fit.isChecked(),
                    show_circles=self.chk_circ.isChecked(),
                    title_fill=self.sp_tfill.value() / 100, title_x=self.sp_tx.value() / 100,
                    title_y=self.sp_ty.value() / 100, title_w_mm=self.sp_tw.value(),
                    title_h_mm=self.sp_th.value(), title_bg=self.chk_tbg.isChecked())

    def export_title(self):
        """Nur das Titelbild speichern (Auswertung/Plots/JJJJ-MM-TT/<name>_title.pdf)."""
        if self.result is None:
            QMessageBox.information(self, "Titelbild", "Noch kein Ergebnis vorhanden.")
            return
        try:
            p = save_figures(self.result, which=["title"], **self._plot_opts())[0]
        except Exception as exc:
            log("Titelbild: " + traceback.format_exc())
            QMessageBox.warning(self, "Titelbild", str(exc))
            return
        self.statusBar().showMessage(
            f"Titelbild gespeichert: {p}   –   LaTeX: \\titleimage*{{\\includegraphics{{{p.name}}}}}"
            "  (mit Stern, sonst bleibt ein weißer Rand)")

    def _fill_table(self, res: FlatTopResult):
        rows = [(k, v) for k, v in res.summary().items() if k != "warnings"]
        self.table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(_pretty_key(k)))
            txt = f"{v:.5g}" if isinstance(v, float) else str(v)
            self.table.setItem(i, 1, QTableWidgetItem(txt))
        self.lbl_warn.setText("\n".join("⚠ " + w for w in res.warnings))

    def export_current(self):
        if self.result is None:
            QMessageBox.information(self, "Export", "Noch kein Ergebnis vorhanden.")
            return
        try:
            paths = save_results(self.result)
            paths += save_figures(self.result, **self._plot_opts())
            paths.append(save_report_for([self.result], self.result.path, self.result.name,
                                         self._plot_opts()))
        except Exception as exc:
            QMessageBox.warning(self, "Export fehlgeschlagen", str(exc))
            return
        self.statusBar().showMessage(f"{len(paths)} Dateien gespeichert in {paths[0].parent}")

    # -- Batch --------------------------------------------------------------
    def start_batch(self):
        if not self.folder:
            QMessageBox.information(self, "Batch", "Bitte zuerst einen Ordner öffnen.")
            return
        files = list_images(self.folder)
        if not files:
            QMessageBox.information(self, "Batch", "Keine Bilder im Ordner.")
            return
        lattice_files = None
        if self.chk_lattice.isChecked():
            lattice_files = self._ask_lattice_files(files)
            if lattice_files is None:
                return          # abgebrochen
        self.batch_thread = BatchThread(self.folder, self.settings(),
                                        self.chk_batch_pdf.isChecked(), files,
                                        self._plot_opts(), lattice_files)
        self.batch_thread.progress.connect(self._on_batch_progress)
        self.batch_thread.done.connect(self._on_batch_done)
        self.batch_thread.failed.connect(lambda m: QMessageBox.warning(self, "Batch", m))
        self.batch_thread.finished.connect(lambda: (self.btn_batch.setEnabled(True),
                                                    self.btn_cancel.setEnabled(False)))
        self.btn_batch.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setRange(0, len(files))
        self.batch_thread.start()

    def _ask_lattice_files(self, files):
        """Fragt, fuer welche Bilder die Gitter-Abschaetzung gerechnet wird.
        Rueckgabe: Menge der Dateinamen oder None bei Abbruch."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Uniformity/Crosstalk im Fallengitter")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(f"Für welche Bilder soll Uniformity/Crosstalk bei Pitch "
                           f"{self.sp_pitch.value():g} µm abgeschätzt werden?\n"
                           f"(Große Profile, die anders ausgewertet werden, abwählen.)"))
        lst = QListWidget()
        for p in files:
            it = QListWidgetItem(p.name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            lst.addItem(it)
        v.addWidget(lst)
        row = QHBoxLayout()
        b_all = QPushButton("Alle"); b_none = QPushButton("Keine")
        b_all.clicked.connect(lambda: [lst.item(i).setCheckState(Qt.Checked) for i in range(lst.count())])
        b_none.clicked.connect(lambda: [lst.item(i).setCheckState(Qt.Unchecked) for i in range(lst.count())])
        row.addWidget(b_all); row.addWidget(b_none); row.addStretch(1)
        v.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        dlg.resize(420, 360)
        self._lattice_dialog = dlg     # fuer Tests erreichbar
        if dlg.exec_() != QDialog.Accepted:
            return None
        return {lst.item(i).text() for i in range(lst.count())
                if lst.item(i).checkState() == Qt.Checked}

    def cancel_batch(self):
        if self.batch_thread:
            self.batch_thread.cancel()

    def _on_batch_progress(self, i, n, name):
        self.progress.setValue(i)
        self.statusBar().showMessage(f"Batch {i}/{n}: {name}")

    def _on_batch_done(self, out: str, n: int):
        self.statusBar().showMessage(f"Batch fertig: {n} Bilder -> {out}")
        QMessageBox.information(self, "Batch fertig", f"{n} Bilder ausgewertet.\n\n{out}")


_PRETTY = {
    "file": "Datei", "um_per_px": "µm pro Pixel (Objekt)", "x0_um": "Zentrum x (µm)",
    "y0_um": "Zentrum y (µm)", "theta_deg": "Winkel lange Achse (°)", "m": "Supergauss-Ordnung m",
    "m_err": "Fehler m", "A": "Amplitude A (counts)", "B": "Offset B (counts)", "R2": "R²",
    "ellipticity_e2": "Elliptizität b/a (1/e²)", "edge_long_um": "Kante 10-90 % lang (µm)",
    "edge_short_um": "Kante 10-90 % kurz (µm)", "D_eq50_data_um": "Äquiv.-Durchm. 50 % (Daten) (µm)",
    "U_plateau_rms_pct": "Plateau-Uniformity σ/µ (%)", "PV_plateau_pct": "Plateau Peak-Valley (%)",
    "plateau_npix": "Pixel im Plateau", "max_value": "Max. Pixelwert (roh)",
    "saturated_pct": "Übersteuert (50 %-Fläche) (%)",
    "D_signal_um": "Signal-Durchmesser inkl. Nebenstruktur (µm)",
    "r_uniform_um": "Radius mit Uniformity unter Schwelle (µm)",
    "uniformity_threshold_pct": "Uniformity-Schwelle (%)",
    "U_atom_pct": "Uniformity atomgewichtet (%)",
    "eta_atom_pct": "Crosstalk η atomgewichtet (%)",
    "sigma_atom_nm": "σ_atom (nm)",
    "pitch_um": "Pitch Fallengitter (µm)",
}


def _pretty_key(k: str) -> str:
    if k in _PRETTY:
        return _PRETTY[k]
    if k.startswith("w_"):
        _, ax, lvl, _u = k.split("_")
        name = {"long": "lange Halbachse", "short": "kurze Halbachse",
                "h": "horizontal", "v": "vertikal"}[ax]
        return f"Radius w {lvl.replace('e2', 'e²')}, {name} (µm)"
    if k.startswith(("U_", "PV_", "eta_")) and k.count("_") == 2:
        q, lvl, _u = k.split("_")
        what = {"U": "Uniformity σ/µ", "PV": "Peak-Valley", "eta": "Crosstalk η"}[q]
        return f"{what} in Region {lvl.replace('e2', 'e²')} (%)"
    if k.startswith("D_"):
        _, ax, lvl, _u = k.split("_")
        return f"Durchmesser {'lang' if ax == 'long' else 'kurz'} {lvl.replace('e2', 'e²')} (µm)"
    return k


def install_excepthook():
    """Unbehandelte Fehler als Meldung zeigen statt den Prozess zu beenden.

    PyQt5 bricht bei einer Exception in einem Slot sonst hart ab
    (Windows: exit code -1073740791 / 0xC0000409).
    """
    def hook(exc_type, exc, tb):
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        log(msg)
        app = QApplication.instance()
        if app is not None:
            box = QMessageBox(QMessageBox.Critical, "Unerwarteter Fehler",
                              f"{exc_type.__name__}: {exc}")
            box.setDetailedText(msg)
            box.exec_()
    sys.excepthook = hook

    # Qt-eigene Meldungen (z. B. "QThread: Destroyed while thread is still running")
    # landen unter Windows sonst unsichtbar im Debug-Kanal
    from PyQt5.QtCore import qInstallMessageHandler

    def qt_handler(mode, ctx, message):
        log(f"Qt: {message}")
    qInstallMessageHandler(qt_handler)


def _version_hint() -> str:
    from .plotting import MPL_VERSION, MPL_RECOMMENDED
    if MPL_VERSION < MPL_RECOMMENDED:
        v = ".".join(map(str, MPL_VERSION))
        return (f"Hinweis: matplotlib {v} ist alt - Plots funktionieren, sehen mit "
                f"'pip install -U matplotlib' (>= 3.7) aber besser aus.")
    return ""


def main(argv=None):
    argv = argv if argv is not None else sys.argv
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    install_excepthook()
    w = MainWindow()
    w.show()
    hint = _version_hint()
    if hint:
        w.statusBar().showMessage(hint)
        print(hint)
    if len(argv) > 1:
        w.load_path(argv[1])          # Datei/Ordner per Kommandozeile
    else:
        QTimer.singleShot(0, w.choose_start_file)   # sonst beim Start auswaehlen
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
