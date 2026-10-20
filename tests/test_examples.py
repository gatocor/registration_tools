"""
Runs the Napari_Flow example pipelines (examples/napari_flow) without the GUI, in a temporary
copy with freshly generated data and a Napari_Flow configuration of its own. Needs napari,
napari_flow_editor, vt-python and SimpleITK; takes about a minute.
"""

import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import registration_tools as rt

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "napari_flow"

try:
    import napari  # noqa: F401
    import SimpleITK  # noqa: F401
    import vt  # noqa: F401
    from napari_flow_editor import plugin_nodes
    READY = True
except ImportError:
    READY = False


def corner_error(estimated, truth):
    """Largest distance (px) between where the corners of the grid land, over all frames."""
    corners = np.array(list(itertools.product(*[(0, n - 1) for n in truth["shape"]])), float)
    corners = np.c_[corners, np.ones(len(corners))]
    return max(np.abs(corners @ a.T - corners @ b.T)[:, :-1].max()
               for a, b in zip(estimated["matrix"], truth["matrix"]))


@unittest.skipUnless(READY, "needs napari, napari_flow_editor, vt-python and SimpleITK")
class TestNapariFlowExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        for path in EXAMPLES.iterdir():
            if path.suffix in (".json", ".csv", ".py"):
                shutil.copy(path, cls.folder)
        subprocess.run([sys.executable, str(cls.folder / "make_example_data.py"),
                        "--output", str(cls.folder / "data")], check=True, capture_output=True)

        config = cls.folder / "config"
        plugin_nodes.write_overlay(plugin_nodes.discover(modules=["registration_tools"]),
                                   config / "plugin_nodes.json")
        cls.env = dict(os.environ, NAPARI_FLOW_CONFIG_DIR=str(config))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_examples(self):
        result = subprocess.run([sys.executable, str(self.folder / "run_examples.py"), "--offline",
                                 "--folder", str(self.folder)], env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:])
        self.assertEqual(result.stdout.count("ok "), 10, result.stdout)

        # 08 saved the drift it found; it should match the motion the data was made with.
        saved = rt.load_registration(self.folder / "registrations" / "drift_2d.npz")
        truth = rt.load_registration(self.folder / "data" / "drift_2d_truth.npz")
        self.assertLess(corner_error(saved, truth), 1.0)

        # 10 saved one registration per CSV row, each the shift of its row.
        for dataset, shift in (("camera", (6.5, -9.25)), ("coins", (-4.0, 7.5)), ("brick", (3.0, 3.0))):
            with self.subTest(dataset=dataset):
                registration = rt.load_registration(self.folder / "registrations" / f"batch_{dataset}.npz")
                np.testing.assert_allclose(registration["matrix"][:2, 2], shift, atol=0.6)


if __name__ == "__main__":
    unittest.main()
