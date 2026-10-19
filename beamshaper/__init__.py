"""BeamShaperPlots - Auswertung von FlatTop-Kamerabildern.

Aufbau (Rechnung, Plots und GUI sind strikt getrennt):
    beamshaper.imageio   Laden von BMP/PNG/TIFF, Bittiefe, Hintergrund
    beamshaper.models    2D-Supergauss-Modell und Radius bei Intensitaets-Level
    beamshaper.analysis  Fit + alle Kennzahlen -> FlatTopResult (kein Plotten)
    beamshaper.plotting  Figures aus einem FlatTopResult (kein Rechnen)
    beamshaper.export    PDF / JSON / CSV schreiben
    beamshaper.batch     ganzen Ordner auswerten -> Summary-CSV
    beamshaper.gui       PyQt5-Oberflaeche
"""
from .analysis import AnalysisSettings, FlatTopResult, analyze_file, analyze_image

__all__ = ["AnalysisSettings", "FlatTopResult", "analyze_file", "analyze_image"]
__version__ = "0.1.0"
