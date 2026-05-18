import os
import sys
import tempfile
import unittest

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
CLI_DIR = os.path.join(REPO_ROOT, "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

import aircraft_diffusion_cfd as cli_module
import rlvr_dataset_bootstrap as rlvr_module


class TestRLVRBootstrap(unittest.TestCase):
    def test_score_candidate_returns_reward_breakdown(self):
        geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
        geometry[2:6, 3:5, 3:5] = 1.0
        design_spec = cli_module.DesignSpec(
            target_speed=42.0,
            wingspan_limit_m=1.8,
            thrust_to_weight_min=0.45,
            turn_rate_min_deg_s=18.0,
            required_static_thrust_n=180.0,
            engine_diameter_mm=140,
            engine_length_mm=260,
            engine_count_min=1,
            engine_count_max=2,
            payload_mass_min_g=500,
            payload_mass_max_g=2000,
            takeoff_distance_min_m=120,
            takeoff_distance_max_m=250,
            wall_thickness_min_mm=1,
            wall_thickness_max_mm=2,
            part_count_min=1,
            part_count_max=8,
            manufacturing_method="foam_core_hotwire",
        )

        reward = rlvr_module.score_candidate(
            geometry=geometry,
            design_spec=design_spec,
            config=rlvr_module.RLVRBootstrapConfig(min_total_reward=0.05),
        )

        self.assertIn("accepted", reward)
        self.assertIn("total_reward", reward)
        self.assertIn("reward_components", reward)
        self.assertIn("connectivity", reward["reward_components"])
        self.assertIn("manufacturing", reward["reward_components"])

    def test_bootstrap_dataset_writes_only_accepted_examples(self):
        good_geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
        good_geometry[2:6, 3:5, 3:5] = 1.0
        bad_geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
        bad_geometry[1, 1, 1] = 1.0
        bad_geometry[6, 6, 6] = 1.0

        good_spec = cli_module.DesignSpec(
            target_speed=42.0,
            wingspan_limit_m=1.8,
            thrust_to_weight_min=0.48,
            turn_rate_min_deg_s=20.0,
            required_static_thrust_n=200.0,
            engine_diameter_mm=145,
            engine_length_mm=280,
            engine_count_min=1,
            engine_count_max=2,
            payload_mass_min_g=500,
            payload_mass_max_g=1800,
            takeoff_distance_min_m=110,
            takeoff_distance_max_m=220,
            wall_thickness_min_mm=1,
            wall_thickness_max_mm=2,
            part_count_min=1,
            part_count_max=6,
            manufacturing_method="fdm_pla_0p4mm",
        )
        bad_spec = cli_module.DesignSpec(
            target_speed=42.0,
            wingspan_limit_m=1.8,
            thrust_to_weight_min=0.28,
            turn_rate_min_deg_s=10.0,
            required_static_thrust_n=90.0,
            engine_diameter_mm=110,
            engine_length_mm=220,
            engine_count_min=2,
            engine_count_max=4,
            payload_mass_min_g=2500,
            payload_mass_max_g=6000,
            takeoff_distance_min_m=350,
            takeoff_distance_max_m=650,
            wall_thickness_min_mm=1,
            wall_thickness_max_mm=1,
            part_count_min=12,
            part_count_max=24,
            manufacturing_method="sheet_balsa_tabbed",
        )

        candidates = [
            {
                "latent": torch.ones(16, dtype=torch.float32),
                "geometry": good_geometry,
                "design_spec": good_spec,
                "condition_vector": cli_module.build_condition_vector(good_spec),
            },
            {
                "latent": torch.zeros(16, dtype=torch.float32),
                "geometry": bad_geometry,
                "design_spec": bad_spec,
                "condition_vector": cli_module.build_condition_vector(bad_spec),
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "bootstrap_dataset.pt")
            summary = rlvr_module.bootstrap_dataset(
                output_path=output_path,
                candidates=candidates,
                config=rlvr_module.RLVRBootstrapConfig(min_total_reward=0.15),
            )

            self.assertTrue(os.path.exists(output_path))
            payload = torch.load(output_path, map_location="cpu")

        self.assertEqual(summary["num_candidates"], 2)
        self.assertEqual(summary["num_accepted"], 1)
        self.assertEqual(payload["geometries"].shape[0], 1)
        self.assertEqual(payload["condition_vectors"].shape[1], cli_module.infer_conditioning_dim())
        self.assertEqual(len(payload["reward_records"]), 1)

    def test_generate_candidate_pool_reuses_conditioned_dataset(self):
        candidates = rlvr_module.generate_candidate_pool(
            num_samples=3,
            grid_size=8,
            latent_dim=16,
            seed=123,
        )

        self.assertEqual(len(candidates), 3)
        self.assertIn("condition_vector", candidates[0])
        self.assertIn("design_spec", candidates[0])
        self.assertEqual(
            candidates[0]["condition_vector"].shape[-1],
            cli_module.infer_conditioning_dim(),
        )


if __name__ == "__main__":
    unittest.main()
