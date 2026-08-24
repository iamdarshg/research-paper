import os
import sys
import unittest


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from config import CFDConfig as ConfigModuleCFDConfig


class TestConfigSourceOfTruth(unittest.TestCase):
    """R9 (PR 41 review, item 9): the flow-condition fields every CFDConfig
    exposes must be sourced from the global config.yaml — the single source of
    truth — not from divergent hardcoded defaults.

    There are two CFDConfig classes (config.py and aircraft_diffusion_cfd.py);
    they must agree on the flow fields the claim-bearing run depends on, and
    both must honor config.yaml for mach_number / reynolds_number /
    simulation_steps."""

    def test_flow_fields_are_sourced_from_global_yaml(self):
        cfg = ConfigModuleCFDConfig(base_grid_resolution=96)
        # Values declared in CLI/config.yaml (the source of truth).
        self.assertEqual(cfg.mach_number, 0.3)
        self.assertEqual(cfg.reynolds_number, 1e6)
        self.assertEqual(cfg.simulation_steps, 1000)

    def test_duplicate_cfdconfig_family_agrees_on_flow_fields(self):
        from aircraft_diffusion_cfd import CFDConfig as TrainerCFDConfig

        trainer_cfg = TrainerCFDConfig(base_grid_resolution=96)
        config_cfg = ConfigModuleCFDConfig(base_grid_resolution=96)
        for field in ("mach_number", "reynolds_number", "simulation_steps"):
            self.assertEqual(
                getattr(trainer_cfg, field),
                getattr(config_cfg, field),
                "the duplicate CFDConfig family silently diverged on %s" % field,
            )


if __name__ == "__main__":
    unittest.main()
