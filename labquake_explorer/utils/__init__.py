"""Utilities package for Labquake Explorer."""

__all__ = ['LabquakeExplorerConfig', 'CohesiveCrack']


def __getattr__(name):
    if name == "LabquakeExplorerConfig":
        from labquake_explorer.utils.config import LabquakeExplorerConfig

        return LabquakeExplorerConfig
    if name == "CohesiveCrack":
        from labquake_explorer.utils.cohesive_crack import CohesiveCrack

        return CohesiveCrack
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
