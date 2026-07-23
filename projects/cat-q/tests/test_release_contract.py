import ast
import tempfile
from pathlib import Path
from unittest import mock
import unittest

import yaml

from main import parse_arguments


ROOT = Path(__file__).resolve().parents[1]


class EvaluationReleaseContractTest(unittest.TestCase):
    def test_full_training_config_is_accepted(self):
        config_path = ROOT / "configs" / "qwen3-4b" / "config.yaml"
        args = parse_arguments(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                "/tmp/parameters.pth",
                "--eval_ppl",
            ]
        )
        self.assertEqual(args.model, "Qwen/Qwen3-4B")
        self.assertEqual(args.wbits, 1)
        self.assertEqual(args.r, 64)
        self.assertTrue(args.learnable_scale)
        self.assertIn("batch_size", args.ignored_config_keys)
        self.assertIn("epochs", args.ignored_config_keys)
        self.assertIn("lora_quant", args.ignored_config_keys)
        self.assertIn("loss_function", args.ignored_config_keys)

    def test_unknown_yaml_fields_are_ignored_for_forward_compatibility(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config:
            yaml.safe_dump({"model": "base/model", "future_training_knob": 7}, config)
            config.flush()
            args = parse_arguments(
                ["--config", config.name, "--checkpoint", "/tmp/parameters.pth", "--tasks", "piqa"]
            )
        self.assertEqual(args.ignored_config_keys, ["future_training_knob"])

    def test_training_cli_flags_are_not_exposed(self):
        for flag in ("--epochs", "--use_ddp", "--loss_function", "--learnable_factor_lr"):
            with self.subTest(flag=flag), mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_arguments(
                        [
                            "--model",
                            "base/model",
                            "--checkpoint",
                            "/tmp/parameters.pth",
                            "--eval_ppl",
                            flag,
                            "1",
                        ]
                    )

    def test_checkpoint_and_action_are_required(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_arguments(["--model", "base/model", "--eval_ppl"])
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_arguments(["--model", "base/model", "--checkpoint", "/tmp/parameters.pth"])

    def test_training_sources_are_absent(self):
        for relative in ("auto_train_ddp.sh", "train_utils.py", "quantize/catq.py"):
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_production_python_has_no_training_runtime(self):
        forbidden = (
            "torch.optim",
            ".backward(",
            "get_scheduler(",
            "DistributedDataParallel",
            "train_one_round",
        )
        for path in ROOT.rglob("*.py"):
            if "tests" in path.parts:
                continue
            source = path.read_text()
            for marker in forbidden:
                with self.subTest(path=path, marker=marker):
                    self.assertNotIn(marker, source)

    def test_all_torch_load_calls_use_restricted_mode(self):
        for path in ROOT.rglob("*.py"):
            if "tests" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                    and node.func.attr == "load"
                ):
                    continue
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                self.assertIn("weights_only", keywords, f"unsafe torch.load in {path}")
                self.assertIs(keywords["weights_only"].value, True)


if __name__ == "__main__":
    unittest.main()
