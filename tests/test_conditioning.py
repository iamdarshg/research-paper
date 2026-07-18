import os
import sys
import types
import unittest
from unittest import mock

import torch
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
CLI_DIR = os.path.join(REPO_ROOT, "CLI")
CONFIG_PATH = os.path.join(REPO_ROOT, "CLI", "conditioning_schema.yaml")
TRAINING_CONFIG_PATH = os.path.join(REPO_ROOT, "CLI", "config.yaml")
README_PATH = os.path.join(
    REPO_ROOT,
    "docs",
    "cheapest-viable-conditioned-generator",
    "README.md",
)

if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

import aircraft_diffusion_cfd as cli_module


def _load_conditioning_schema():
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _expected_vector_layout(schema):
    layout = [feature["name"] for feature in schema["scalar_features"]]
    for feature_name, feature_schema in schema["categorical_features"].items():
        layout.extend(
            f"{feature_name}__{category}" for category in feature_schema["categories"]
        )
    return layout


def _vectorize_condition(schema, sample):
    values = []

    for feature in schema["scalar_features"]:
        values.append(float(sample[feature["name"]]))

    for feature_name, feature_schema in schema["categorical_features"].items():
        categories = feature_schema["categories"]
        selected = sample[feature_name]
        if selected not in categories:
            raise ValueError(
                f"Unknown category {selected!r} for {feature_name}; "
                f"expected one of {categories!r}"
            )
        values.extend(1.0 if selected == category else 0.0 for category in categories)

    tensor_dtype = getattr(torch, schema["tensor_dtype"])
    return torch.tensor(values, dtype=tensor_dtype)


class TestConditioningSchema(unittest.TestCase):
    def test_conditioning_schema_has_stable_layout_and_defaults(self):
        schema = _load_conditioning_schema()
        expected_layout = _expected_vector_layout(schema)

        self.assertEqual(schema["vector_layout"], expected_layout)
        self.assertEqual(schema["vector_dim"], len(expected_layout))

        manufacturing = schema["categorical_features"]["manufacturing_method"]
        self.assertEqual(
            manufacturing["categories"],
            [
                "foam_core_hotwire",
                "fdm_pla_0p4mm",
                "fdm_pla_0p6mm",
                "sheet_balsa_tabbed",
                "composite_wet_layup",
            ],
        )
        self.assertEqual(manufacturing["default"], "fdm_pla_0p4mm")

        for feature in schema["scalar_features"]:
            self.assertIn("name", feature)
            self.assertIn("default", feature)

        for feature_name, feature_schema in schema["categorical_features"].items():
            self.assertIn(feature_schema["default"], feature_schema["categories"])
            self.assertEqual(
                len(feature_schema["categories"]),
                len(set(feature_schema["categories"])),
                msg=f"{feature_name} categories should stay unique",
            )

    def test_conditioning_schema_vectorizes_small_metadata_payload(self):
        schema = _load_conditioning_schema()
        payload = {
            "target_speed_mps": 42.0,
            "wingspan_limit_m": 1.8,
            "thrust_to_weight_min": 0.45,
            "turn_rate_min_deg_s": 18.0,
            "required_static_thrust_n": 180.0,
            "engine_diameter_mm": 140,
            "engine_length_mm": 260,
            "engine_count_min": 1,
            "engine_count_max": 2,
            "payload_mass_min_g": 500,
            "payload_mass_max_g": 2000,
            "takeoff_distance_min_m": 120,
            "takeoff_distance_max_m": 250,
            "wall_thickness_min_mm": 1,
            "wall_thickness_max_mm": 2,
            "part_count_min": 1,
            "part_count_max": 8,
            "manufacturing_method": "fdm_pla_0p6mm",
        }

        vector = _vectorize_condition(schema, payload)
        self.assertEqual(vector.shape, (schema["vector_dim"],))
        self.assertEqual(vector.dtype, torch.float32)

        layout = schema["vector_layout"]
        self.assertEqual(vector[layout.index("target_speed_mps")].item(), 42.0)
        self.assertAlmostEqual(
            vector[layout.index("wingspan_limit_m")].item(),
            1.8,
            places=6,
        )
        self.assertAlmostEqual(
            vector[layout.index("thrust_to_weight_min")].item(),
            0.45,
            places=6,
        )
        self.assertAlmostEqual(
            vector[layout.index("turn_rate_min_deg_s")].item(),
            18.0,
            places=6,
        )
        self.assertEqual(vector[layout.index("required_static_thrust_n")].item(), 180.0)
        self.assertEqual(vector[layout.index("engine_diameter_mm")].item(), 140.0)
        self.assertEqual(vector[layout.index("engine_length_mm")].item(), 260.0)
        self.assertEqual(vector[layout.index("engine_count_min")].item(), 1.0)
        self.assertEqual(vector[layout.index("engine_count_max")].item(), 2.0)
        self.assertEqual(vector[layout.index("payload_mass_min_g")].item(), 500.0)
        self.assertEqual(vector[layout.index("payload_mass_max_g")].item(), 2000.0)
        self.assertEqual(vector[layout.index("takeoff_distance_min_m")].item(), 120.0)
        self.assertEqual(vector[layout.index("takeoff_distance_max_m")].item(), 250.0)
        self.assertEqual(vector[layout.index("wall_thickness_min_mm")].item(), 1.0)
        self.assertEqual(vector[layout.index("wall_thickness_max_mm")].item(), 2.0)
        self.assertEqual(vector[layout.index("part_count_min")].item(), 1.0)
        self.assertEqual(vector[layout.index("part_count_max")].item(), 8.0)
        self.assertEqual(
            vector[layout.index("manufacturing_method__fdm_pla_0p6mm")].item(),
            1.0,
        )
        self.assertEqual(
            vector[layout.index("manufacturing_method__foam_core_hotwire")].item(),
            0.0,
        )
        self.assertEqual(
            vector[layout.index("manufacturing_method__sheet_balsa_tabbed")].item(),
            0.0,
        )

        categorical_feature_count = len(schema["categorical_features"])
        categorical_slice = vector[len(schema["scalar_features"]) :]
        self.assertEqual(categorical_slice.sum().item(), float(categorical_feature_count))

    def test_conditioning_readme_mentions_documented_tensor_contract(self):
        schema = _load_conditioning_schema()
        with open(README_PATH, "r", encoding="utf-8") as handle:
            readme = handle.read()

        self.assertIn(f"[batch, {schema['vector_dim']}]", readme)
        self.assertIn("partial structured conditioning path", readme)
        self.assertIn("public CLI currently exposes only a subset", readme)
        with open(TRAINING_CONFIG_PATH, "r", encoding="utf-8") as handle:
            training_config = yaml.safe_load(handle)
        self.assertEqual(
            training_config["conditioning"]["schema_path"],
            "conditioning_schema.yaml",
        )

        manufacturing_categories = schema["categorical_features"]["manufacturing_method"][
            "categories"
        ]
        for category in manufacturing_categories:
            self.assertIn(category, readme)


class TestLiveConditioningPath(unittest.TestCase):
    def test_invalid_design_spec_combinations_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "engine_count_min"):
            cli_module.DesignSpec(engine_count_min=3, engine_count_max=1)

        with self.assertRaisesRegex(ValueError, "payload_mass_min_g"):
            cli_module.DesignSpec(payload_mass_min_g=2500, payload_mass_max_g=500)

        with self.assertRaisesRegex(ValueError, "takeoff_distance_min_m"):
            cli_module.DesignSpec(takeoff_distance_min_m=300, takeoff_distance_max_m=150)

        with self.assertRaisesRegex(ValueError, "wall_thickness_min_mm"):
            cli_module.DesignSpec(wall_thickness_min_mm=3, wall_thickness_max_mm=1)

        with self.assertRaisesRegex(ValueError, "part_count_min"):
            cli_module.DesignSpec(part_count_min=9, part_count_max=4)

        with self.assertRaisesRegex(ValueError, "required_static_thrust_n"):
            cli_module.DesignSpec(required_static_thrust_n=0.0)

        with self.assertRaisesRegex(ValueError, "manufacturing_method"):
            cli_module.DesignSpec(manufacturing_method="mystery_process")

    def test_build_condition_vector_matches_documented_schema(self):
        schema = _load_conditioning_schema()
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
            manufacturing_method="fdm_pla_0p6mm",
        )

        vector = cli_module.build_condition_vector(design_spec)
        self.assertEqual(vector.shape, (schema["vector_dim"],))
        self.assertEqual(vector.dtype, torch.float32)

        layout = schema["vector_layout"]
        self.assertEqual(vector[layout.index("target_speed_mps")].item(), 42.0)
        self.assertAlmostEqual(vector[layout.index("wingspan_limit_m")].item(), 1.8, places=6)
        self.assertAlmostEqual(vector[layout.index("thrust_to_weight_min")].item(), 0.45, places=6)
        self.assertAlmostEqual(vector[layout.index("turn_rate_min_deg_s")].item(), 18.0, places=6)
        self.assertEqual(vector[layout.index("required_static_thrust_n")].item(), 180.0)
        self.assertEqual(vector[layout.index("engine_diameter_mm")].item(), 140.0)
        self.assertEqual(vector[layout.index("engine_length_mm")].item(), 260.0)
        self.assertEqual(vector[layout.index("engine_count_min")].item(), 1.0)
        self.assertEqual(vector[layout.index("engine_count_max")].item(), 2.0)
        self.assertEqual(vector[layout.index("payload_mass_min_g")].item(), 500.0)
        self.assertEqual(vector[layout.index("payload_mass_max_g")].item(), 2000.0)
        self.assertEqual(vector[layout.index("takeoff_distance_min_m")].item(), 120.0)
        self.assertEqual(vector[layout.index("takeoff_distance_max_m")].item(), 250.0)
        self.assertEqual(vector[layout.index("wall_thickness_min_mm")].item(), 1.0)
        self.assertEqual(vector[layout.index("wall_thickness_max_mm")].item(), 2.0)
        self.assertEqual(vector[layout.index("part_count_min")].item(), 1.0)
        self.assertEqual(vector[layout.index("part_count_max")].item(), 8.0)
        self.assertEqual(
            vector[layout.index("manufacturing_method__fdm_pla_0p6mm")].item(),
            1.0,
        )

    def test_normalize_condition_vector_scales_scalar_slots_only(self):
        schema = _load_conditioning_schema()
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
            manufacturing_method="fdm_pla_0p6mm",
        )

        vector = cli_module.build_condition_vector(design_spec)
        normalized = cli_module.normalize_condition_vector_tensor(vector.unsqueeze(0)).squeeze(0)
        layout = schema["vector_layout"]

        self.assertAlmostEqual(normalized[layout.index("target_speed_mps")].item(), 0.42, places=6)
        self.assertAlmostEqual(normalized[layout.index("payload_mass_max_g")].item(), 0.2, places=6)
        self.assertAlmostEqual(normalized[layout.index("required_static_thrust_n")].item(), 0.18, places=6)
        self.assertEqual(
            normalized[layout.index("manufacturing_method__fdm_pla_0p6mm")].item(),
            1.0,
        )

    def test_dataset_emits_condition_vector_and_design_spec(self):
        schema = _load_conditioning_schema()
        dataset = cli_module.AircraftDesignDataset(
            num_samples=4,
            grid_size=8,
            seed=123,
            latent_dim=16,
        )

        sample = dataset[0]
        self.assertIn("condition_vector", sample)
        self.assertIn("design_spec", sample)
        self.assertEqual(sample["condition_vector"].shape, (schema["vector_dim"],))
        self.assertIsInstance(sample["design_spec"], cli_module.DesignSpec)

    def test_sample_design_spec_respects_numeric_bounds(self):
        rng = __import__("random").Random(7)
        for _ in range(32):
            sample = cli_module.sample_design_spec(rng)
            self.assertLessEqual(sample.engine_count_min, sample.engine_count_max)
            self.assertLessEqual(sample.payload_mass_min_g, sample.payload_mass_max_g)
            self.assertLessEqual(sample.takeoff_distance_min_m, sample.takeoff_distance_max_m)
            self.assertLessEqual(sample.wall_thickness_min_mm, sample.wall_thickness_max_mm)
            self.assertLessEqual(sample.part_count_min, sample.part_count_max)

    def test_generator_passes_condition_vector_to_consistency_model(self):
        schema = _load_conditioning_schema()
        generator = object.__new__(cli_module.OptimizedAircraftGenerator)
        generator.device = torch.device("cpu")
        generator.model_config = types.SimpleNamespace(latent_dim=8)
        generator.consistency_model = mock.Mock()
        generator.consistency_model.fast_inference.return_value = torch.ones((1, 8))
        generator.converter = mock.Mock(return_value=torch.zeros((1, 8, 8, 8)))

        design_spec = cli_module.DesignSpec(
            target_speed=55.0,
            wingspan_limit_m=1.6,
            thrust_to_weight_min=0.52,
            turn_rate_min_deg_s=21.0,
            required_static_thrust_n=220.0,
            engine_diameter_mm=155,
            engine_length_mm=300,
            engine_count_min=1,
            engine_count_max=2,
            payload_mass_min_g=750,
            payload_mass_max_g=1800,
            takeoff_distance_min_m=100,
            takeoff_distance_max_m=180,
            wall_thickness_min_mm=1,
            wall_thickness_max_mm=3,
            part_count_min=2,
            part_count_max=10,
            manufacturing_method="foam_core_hotwire",
        )
        cli_module.OptimizedAircraftGenerator.generate(generator, design_spec, num_steps=3)

        fast_inference_kwargs = generator.consistency_model.fast_inference.call_args.kwargs
        self.assertIn("condition", fast_inference_kwargs)
        self.assertEqual(
            fast_inference_kwargs["condition"].shape[-1],
            schema["vector_dim"],
        )

    def test_generator_uses_zero_condition_for_unconditioned_checkpoint(self):
        schema = _load_conditioning_schema()
        generator = object.__new__(cli_module.OptimizedAircraftGenerator)
        generator.device = torch.device("cpu")
        generator.model_config = types.SimpleNamespace(latent_dim=8)
        generator.consistency_model = mock.Mock()
        generator.consistency_model.fast_inference.return_value = torch.ones((1, 8))
        generator.converter = mock.Mock(return_value=torch.zeros((1, 8, 8, 8)))

        cli_module.OptimizedAircraftGenerator.generate(
            generator,
            None,
            num_steps=3,
        )

        condition = generator.consistency_model.fast_inference.call_args.kwargs[
            "condition"
        ]
        self.assertEqual(condition.shape, (1, schema["vector_dim"]))
        self.assertEqual(float(condition.abs().sum()), 0.0)

    def test_condition_response_smoke_summary_reports_meaningful_deltas(self):
        with mock.patch.object(cli_module.Path, "mkdir"), \
             mock.patch.object(cli_module.Path, "open", mock.mock_open()) as mock_file:
            summary = cli_module.generate_condition_response_smoke_summary(
                output_path=os.path.join(REPO_ROOT, "build", "condition_response_smoke.json"),
                grid_size=16,
                latent_dim=16,
                seed=0,
            )

        self.assertEqual(summary["mode"], "condition-response smoke only")
        self.assertEqual(len(summary["cases"]), 2)
        self.assertGreater(summary["deltas"]["occupancy_ratio"], 0.0)
        self.assertGreater(summary["deltas"]["span_y_fraction"], 0.0)
        self.assertGreater(summary["deltas"]["engine_proxy"], 0.0)
        handle = mock_file()
        handle.write.assert_called()


if __name__ == "__main__":
    unittest.main()
