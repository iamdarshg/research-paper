import os
import sys
import unittest


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_diffusion_cfd import resolve_grounded_grid_size
from build_grounded_aircraft_corpus import validate_cfd_outputs


class TestGroundedContracts(unittest.TestCase):
    def test_resolve_grounded_grid_size_uses_detected_manifest_grid(self):
        resolved = resolve_grounded_grid_size(
            None,
            detected_grid_size=32,
            solver="D3Q27",
            source_label="manifest.jsonl",
        )

        self.assertEqual(resolved, 32)

    def test_resolve_grounded_grid_size_rejects_explicit_mismatch(self):
        with self.assertRaises(ValueError):
            resolve_grounded_grid_size(
                16,
                detected_grid_size=32,
                solver="D3Q27",
                source_label="manifest.jsonl",
            )

    def test_validate_cfd_outputs_accepts_finite_positive_drag(self):
        validate_cfd_outputs(
            source_id="demo",
            grid_size=32,
            steps=20,
            cfd={"drag_coefficient": 1.25, "lift_coefficient": 0.08},
        )

    def test_validate_cfd_outputs_rejects_negative_drag(self):
        with self.assertRaises(ValueError):
            validate_cfd_outputs(
                source_id="demo",
                grid_size=32,
                steps=20,
                cfd={"drag_coefficient": -0.1, "lift_coefficient": 0.08},
            )

    def test_validate_cfd_outputs_rejects_nonfinite_drag(self):
        with self.assertRaises(ValueError):
            validate_cfd_outputs(
                source_id="demo",
                grid_size=32,
                steps=20,
                cfd={"drag_coefficient": float("nan"), "lift_coefficient": 0.08},
            )


if __name__ == "__main__":
    unittest.main()
