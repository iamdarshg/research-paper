import os
import sys
import unittest

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI"))

from raw_solver_diagnostics import (
    DEFAULT_RAW_PROFILE_FIELDS,
    build_shock_tube_initial_fields,
    line_profile,
    summarize_profile,
)


class RawSolverDiagnosticsTests(unittest.TestCase):
    def test_shock_tube_initial_fields_preserve_raw_left_right_states(self):
        fields = build_shock_tube_initial_fields(
            shape=(8, 3, 2),
            left_density=1.0,
            right_density=0.125,
            left_pressure=1.0,
            right_pressure=0.1,
            gas_constant=1.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(fields["density"].shape), (8, 3, 2))
        self.assertTrue(torch.allclose(fields["density"][:4], torch.full((4, 3, 2), 1.0)))
        self.assertTrue(torch.allclose(fields["density"][4:], torch.full((4, 3, 2), 0.125)))
        self.assertTrue(torch.allclose(fields["temperature"][:4], torch.full((4, 3, 2), 1.0)))
        self.assertTrue(torch.allclose(fields["temperature"][4:], torch.full((4, 3, 2), 0.8)))
        self.assertTrue(torch.allclose(fields["pressure"][:4], torch.full((4, 3, 2), 1.0)))
        self.assertTrue(torch.allclose(fields["pressure"][4:], torch.full((4, 3, 2), 0.1)))

    def test_line_profile_reports_raw_fields_without_normalized_aliases(self):
        density = torch.arange(4, dtype=torch.float32).view(4, 1, 1).expand(4, 2, 2) + 1.0
        fields = {
            "density": density,
            "temperature": torch.full_like(density, 300.0),
            "pressure": density * 300.0,
            "flow_pressure_lu": density / 3.0,
            "ux_lattice": density * 0.01,
            "shock_sensor": torch.zeros_like(density),
        }

        rows = line_profile(fields, case_name="unit")
        first = rows[0]

        self.assertEqual(set(DEFAULT_RAW_PROFILE_FIELDS).issubset(first.keys()), True)
        self.assertEqual(first["case"], "unit")
        self.assertEqual(first["x_index"], 0)
        self.assertAlmostEqual(first["density_mean"], 1.0)
        self.assertAlmostEqual(first["pressure_mean"], 300.0)
        self.assertFalse(any("normalized" in key.lower() for key in first))

    def test_summarize_profile_uses_raw_min_max_and_ratio(self):
        rows = [
            {"pressure_mean": 1.0, "density_mean": 2.0, "temperature_mean": 3.0, "ux_lattice_mean": 0.1},
            {"pressure_mean": 0.25, "density_mean": 1.0, "temperature_mean": 4.0, "ux_lattice_mean": -0.2},
        ]

        summary = summarize_profile(rows)

        self.assertEqual(summary["pressure_min"], 0.25)
        self.assertEqual(summary["pressure_max"], 1.0)
        self.assertEqual(summary["pressure_max_to_min_ratio"], 4.0)
        self.assertEqual(summary["density_min"], 1.0)
        self.assertEqual(summary["temperature_max"], 4.0)
        self.assertEqual(summary["ux_lattice_min"], -0.2)


if __name__ == "__main__":
    unittest.main()
