"""Data handling package for Labquake Explorer."""

__all__ = ['DataManager', 'FileHandler', 'EventProcessor']


def __getattr__(name):
    if name == "DataManager":
        from labquake_explorer.data.data_manager import DataManager

        return DataManager
    if name == "FileHandler":
        from labquake_explorer.data.file_handler import FileHandler

        return FileHandler
    if name == "EventProcessor":
        from labquake_explorer.data.event_processor import EventProcessor

        return EventProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
