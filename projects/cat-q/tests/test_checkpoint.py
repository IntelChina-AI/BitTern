import tempfile
from pathlib import Path
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised in minimal documentation environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CheckpointTest(unittest.TestCase):
    def test_plain_checkpoint_round_trip(self):
        from quantize.checkpoint import load_catq_parameters

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "parameters.pth"
            expected = {0: {"scale": torch.tensor([1.0, 2.0])}}
            torch.save(expected, path)
            actual = load_catq_parameters(path)
            self.assertTrue(torch.equal(actual[0]["scale"], expected[0]["scale"]))

    def test_low_cpu_manifest_round_trip(self):
        from quantize.checkpoint import CATQ_PARAMETERS_MANIFEST_FORMAT, load_catq_parameters

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            layer_dir = root / "parameters_layers"
            layer_dir.mkdir()
            layer_path = layer_dir / "layer_0.pth"
            expected = {"smooth_scale": torch.tensor([3.0])}
            torch.save(expected, layer_path)

            manifest_path = root / "parameters.pth"
            torch.save(
                {
                    "__format__": CATQ_PARAMETERS_MANIFEST_FORMAT,
                    "layers": {0: "parameters_layers/layer_0.pth"},
                },
                manifest_path,
            )

            store = load_catq_parameters(manifest_path)
            self.assertIn(0, store)
            self.assertTrue(torch.equal(store[0]["smooth_scale"], expected["smooth_scale"]))

    def test_manifest_cannot_escape_checkpoint_directory(self):
        from quantize.checkpoint import CATQ_PARAMETERS_MANIFEST_FORMAT, load_catq_parameters

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "parameters.pth"
            torch.save(
                {
                    "__format__": CATQ_PARAMETERS_MANIFEST_FORMAT,
                    "layers": {0: "../outside.pth"},
                },
                manifest_path,
            )
            with self.assertRaises(ValueError):
                load_catq_parameters(manifest_path)
