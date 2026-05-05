"""Event processing for Labquake Explorer."""

from __future__ import annotations

import numpy as np
from labquake_explorer.utils import tpc5
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class EventProcessor:
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path

    def set_data_path(self, data_path: Path) -> None:
        """Set the base path for resolving relative file paths"""
        self.data_path = data_path

    def extract_events(self, run_data: Dict[str, Any], event_indices: List[int], window: float) -> List[Dict]:
        """Extract events from run data using provided indices and time window
        
        Args:
            run_data: Dictionary containing run data including strain data
            event_indices: List of indices marking event locations
            window: Time window size (in seconds) before and after each event
            
        Returns:
            List of extracted event dictionaries
        """
        events = []
        for i, idx in enumerate(event_indices):
            event = {}
            event_time = run_data["time"][idx]
            
            # Extract time window around event
            idx_beg = np.argmin(np.abs(event_time - window - run_data["time"]))
            idx_end = np.argmin(np.abs(event_time + window - run_data["time"]))
            
            # Store basic event info
            event['event_time'] = event_time
            event['time'] = run_data['time'][idx_beg:idx_end + 1]

            try:
                # Store mechanical data
                mechanical_fields = [
                    'normal_stress', 'shear_stress', 'friction',
                    'LP_displacement', 'LP_velocity', 'displacement',
                    'normal_displacement', 'acceleration_g', 'acceleration_m_per_s2'
                ]
                
                for field in mechanical_fields:
                    if field in run_data:
                        event[field] = run_data[field][idx_beg:idx_end + 1]

                if 'raw' in run_data and isinstance(run_data['raw'], dict):
                    event['raw'] = {
                        key: np.asarray(value)[idx_beg:idx_end + 1]
                        for key, value in run_data['raw'].items()
                    }

                # Handle strain data if available
                if 'strain' in run_data:
                    event['strain'] = self._process_strain_data(
                        run_data, event_time, window, idx_beg, idx_end
                    )

            except Exception as e:
                print(f"Warning: Error processing event {i}: {str(e)}")
                # Fallback: store all available array data for this index
                for key in run_data:
                    if key == "events":
                        continue
                    if isinstance(run_data[key], (np.ndarray, list)):
                        try:
                            event[key] = run_data[key][idx_beg:idx_end]
                        except IndexError:
                            event[key] = run_data[key][idx]
            
            events.append(event)
            
        return events

    def _resolve_strain_file(self, relative_path: str) -> Path:
        """Resolve a strain file path stored inside the loaded data."""
        if self.data_path is None:
            raise ValueError("Data path not set. Call set_data_path() first.")

        candidate = Path(relative_path)
        if candidate.is_absolute():
            return candidate

        search_roots = [self.data_path.parent, *self.data_path.parents, Path.cwd()]
        seen: set[Path] = set()
        for root in search_roots:
            if root in seen:
                continue
            seen.add(root)
            resolved = root / candidate
            if resolved.exists():
                return resolved

        raise FileNotFoundError(
            f"Could not resolve strain file '{relative_path}' from '{self.data_path}'."
        )

    def _load_original_strain_window(
        self,
        run_data: Dict[str, Any],
        time_before: float,
        time_after: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load the full-resolution strain window for a single event."""
        strain_data = run_data["strain"]

        if "original" in strain_data:
            original = strain_data["original"]
            ts = np.asarray(original["time"])
            y = np.asarray(original["raw"])
            idx_before = int(np.argmin(np.abs(ts - time_before)))
            idx_after = int(np.argmin(np.abs(ts - time_after)))
            if idx_after < idx_before:
                idx_before, idx_after = idx_after, idx_before
            return ts[idx_before:idx_after + 1], y[:, idx_before:idx_after + 1]

        import h5py

        strain_file = self._resolve_strain_file(strain_data['filename'])
        with h5py.File(strain_file, 'r') as f:
            n_channels = tpc5.getNChannels(f)
            n_samples = tpc5.getNSamples(f, 1)
            trigger_sample = tpc5.getTriggerSample(f, 1, 1)
            sampling_rate = tpc5.getSampleRate(f, 1, 1)
            
            # Calculate time series
            start_time = -trigger_sample / sampling_rate
            end_time = (n_samples - trigger_sample) / sampling_rate
            ts = np.arange(start_time, end_time, 1/sampling_rate)
            ts += strain_data.get('time_offset', 0.0) + run_data['time'][0] - ts[0]

            # Get indices for time window
            idx_before = np.argmin(np.abs(ts - time_before))
            idx_after = np.argmin(np.abs(ts - time_after))
            if idx_after < idx_before:
                idx_before, idx_after = idx_after, idx_before
            tt = ts[idx_before:idx_after + 1]

            # Extract strain data
            y = np.zeros((n_channels, len(tt)))
            for j in range(n_channels):
                y[j, :] = tpc5.getVoltageData(f, j + 1)[idx_before:idx_after + 1]
                baseline_width = max(1, int(y.shape[1] / 100))
                y[j, :] -= y[j, :baseline_width].mean()

        return tt, y

    def _process_strain_data(
        self,
        run_data: Dict[str, Any],
        event_time: float,
        window: float,
        idx_beg: int,
        idx_end: int,
    ) -> Dict[str, Any]:
        """Process strain data for a single event."""
        strain_data = run_data["strain"]
        time_before = event_time - window
        time_after = event_time + window

        tt, y = self._load_original_strain_window(run_data, time_before, time_after)

        strain_time = np.asarray(strain_data['time'])
        idx_before_strain = int(np.argmin(np.abs(strain_time - time_before)))
        idx_after_strain = int(np.argmin(np.abs(strain_time - time_after)))
        if idx_after_strain < idx_before_strain:
            idx_before_strain, idx_after_strain = idx_after_strain, idx_before_strain
        strain_slice = slice(idx_before_strain, idx_after_strain + 1)

        event_strain = {
            'filename_downsampled': strain_data.get('filename_downsampled', ''),
            'filename': strain_data['filename'],
            'time_offset': strain_data.get('time_offset', 0.0),
            'time': run_data['time'][0] + strain_data.get('time_offset', 0.0) + strain_time[strain_slice],
            'raw': np.asarray(strain_data['raw'])[:, strain_slice],
            'original': {
                'time': tt,
                'raw': y,
            },
        }

        for key in ('enabled_channels', 'fitting_channels', 'labels', 'locations', 'stress_label'):
            if key in strain_data:
                event_strain[key] = strain_data[key]

        if 'on_fault_shear_stress_mpa' in strain_data:
            event_strain['on_fault_shear_stress_mpa'] = {
                key: np.asarray(value)[strain_slice]
                for key, value in strain_data['on_fault_shear_stress_mpa'].items()
            }

        return event_strain

    def get_data_at_path(self, data: Dict[str, Any], path: str) -> Any:
        """Get data at specified path"""
        current = data
        for key in path.split('/'):
            if key[0] == '[' and key[-1] == ']':
                key = int(key[1:-1])
            current = current[key]
        return current

    def set_data_at_path(self, data: Dict[str, Any], path: str, value: Any, add_key: bool = False) -> None:
        """Set data at specified path"""
        parts = path.split('/')
        current = data
        
        for i, part in enumerate(parts[:-1]):
            if part[0] == '[' and part[-1] == ']':
                part = int(part[1:-1])
            
            if part not in current and add_key:
                current[part] = {} if i < len(parts) - 2 else None
            current = current[part]
            
        last_key = parts[-1]
        if last_key[0] == '[' and last_key[-1] == ']':
            last_key = int(last_key[1:-1])
        current[last_key] = value
