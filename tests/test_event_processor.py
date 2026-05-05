import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from labquake_explorer.data.event_processor import EventProcessor
from labquake_explorer.data import event_processor as event_processor_module


class _FakeH5File:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False


class EventProcessorTests(unittest.TestCase):
    def test_extract_events_uses_embedded_original_without_hdf5(self):
        processor = EventProcessor(Path("/tmp/example.npz"))
        run_time = np.linspace(0.0, 1.0, 11)
        embedded_raw = np.vstack([np.arange(11), np.arange(11) * 2])

        run_data = {
            "time": run_time,
            "shear_stress": np.linspace(1.0, 2.0, 11),
            "friction": np.linspace(0.1, 0.2, 11),
            "LP_displacement": np.linspace(0.0, 10.0, 11),
            "LP_velocity": np.linspace(0.0, 1.0, 11),
            "displacement": np.linspace(0.0, 0.5, 11),
            "strain": {
                "filename": "unused.tpc5",
                "filename_downsampled": "unused.npz",
                "time_offset": 0.0,
                "time": run_time,
                "raw": embedded_raw,
                "original": {
                    "time": run_time,
                    "raw": embedded_raw,
                },
                "enabled_channels": [True, False],
                "fitting_channels": [True, False],
            },
        }

        event = processor.extract_events(run_data, [5], 0.2)[0]

        self.assertEqual(event["time"].shape[0], 5)
        self.assertEqual(event["strain"]["raw"].shape, (2, 5))
        self.assertEqual(event["strain"]["original"]["raw"].shape, (2, 5))
        self.assertEqual(event["strain"]["enabled_channels"], [True, False])
        np.testing.assert_allclose(event["strain"]["original"]["time"], np.linspace(0.3, 0.7, 5))

    def test_extract_events_resolves_project_relative_strain_file(self):
        with TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            npz_path = workspace / "data" / "npz" / "t0135.npz"
            strain_path = workspace / "data" / "t0135.exp" / "data" / "t0135-4MPa.tpc5"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            strain_path.parent.mkdir(parents=True, exist_ok=True)
            npz_path.touch()
            strain_path.touch()

            processor = EventProcessor(npz_path)
            run_time = np.linspace(0.0, 1.0, 11)
            run_raw = {
                "A5": np.linspace(10.0, 20.0, 11),
                "A6": np.linspace(30.0, 40.0, 11),
            }
            strain_raw = np.vstack([np.arange(11), np.arange(11) * 2])
            stress_series = {
                "B1": np.linspace(0.0, 1.0, 11),
                "B3": np.linspace(1.0, 2.0, 11),
            }

            run_data = {
                "time": run_time,
                "normal_stress": np.linspace(1.0, 2.0, 11),
                "shear_stress": np.linspace(3.0, 4.0, 11),
                "friction": np.linspace(0.1, 0.2, 11),
                "LP_displacement": np.linspace(0.0, 10.0, 11),
                "LP_velocity": np.linspace(0.0, 1.0, 11),
                "displacement": np.linspace(0.0, 0.5, 11),
                "normal_displacement": np.linspace(0.0, 0.05, 11),
                "raw": run_raw,
                "strain": {
                    "filename": "data/t0135.exp/data/t0135-4MPa.tpc5",
                    "filename_downsampled": "data/t0135.exp/data/t0135-4MPa.tpc5",
                    "time_offset": 0.0,
                    "time": run_time,
                    "raw": strain_raw,
                    "enabled_channels": [True, True],
                    "fitting_channels": [True, False],
                    "labels": np.array(["B1", "B3"], dtype=object),
                    "locations": [10.0, 20.0],
                    "stress_label": "on fault shear stress",
                    "on_fault_shear_stress_mpa": stress_series,
                },
            }

            fake_h5py = types.SimpleNamespace(File=lambda *args, **kwargs: _FakeH5File())
            with patch.dict(sys.modules, {"h5py": fake_h5py}):
                with patch.object(event_processor_module.tpc5, "getNChannels", return_value=2):
                    with patch.object(event_processor_module.tpc5, "getNSamples", return_value=11):
                        with patch.object(event_processor_module.tpc5, "getTriggerSample", return_value=0):
                            with patch.object(event_processor_module.tpc5, "getSampleRate", return_value=10):
                                with patch.object(
                                    event_processor_module.tpc5,
                                    "getVoltageData",
                                    side_effect=[np.arange(11), np.arange(11) * 2],
                                ):
                                    event = processor.extract_events(run_data, [5], 0.2)[0]

            self.assertEqual(event["time"].shape[0], 5)
            self.assertEqual(event["raw"]["A5"].shape[0], 5)
            self.assertEqual(event["strain"]["raw"].shape, (2, 5))
            self.assertEqual(event["strain"]["original"]["raw"].shape, (2, 5))
            self.assertEqual(event["strain"]["locations"], [10.0, 20.0])
            self.assertEqual(event["strain"]["fitting_channels"], [True, False])
            self.assertEqual(sorted(event["strain"]["on_fault_shear_stress_mpa"].keys()), ["B1", "B3"])


if __name__ == "__main__":
    unittest.main()
