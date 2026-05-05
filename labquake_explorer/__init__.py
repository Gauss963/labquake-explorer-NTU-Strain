"""Labquake Explorer package."""

__version__ = "0.1.0"
__all__ = ["LabquakeExplorer", "DataManager"]


def __getattr__(name):
    if name == "LabquakeExplorer":
        from labquake_explorer.ui.labquake_explorer import LabquakeExplorer

        return LabquakeExplorer
    if name == "DataManager":
        from labquake_explorer.data.data_manager import DataManager

        return DataManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
