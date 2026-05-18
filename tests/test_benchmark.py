import io
import json
import tempfile
import unittest
import shutil
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

import run_internal_benchmark as benchmark


class TestBenchmarkDiscovery(unittest.TestCase):
    def test_discover_root_stls_prioritizes_20mm_cube(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '20mm_cube.stl').write_text('solid cube\nendsolid cube\n')
            (root / 'alpha.stl').write_text('solid alpha\nendsolid alpha\n')
            nested = root / 'nested'
            nested.mkdir()
            (nested / 'ignored.stl').write_text('solid ignored\nendsolid ignored\n')

            stls = benchmark.discover_root_stls(root)

            self.assertEqual([p.name for p in stls], ['20mm_cube.stl', 'alpha.stl'])
            self.assertTrue(all(p.parent == root.resolve() for p in stls))

    def test_discover_stls_supports_recursive_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '20mm_cube.stl').write_text('solid cube\nendsolid cube\n')
            nested = root / 'nested'
            nested.mkdir()
            (nested / 'alpha.stl').write_text('solid alpha\nendsolid alpha\n')

            root_only = benchmark.discover_stls(root)
            recursive = benchmark.discover_stls(root, recursive=True)

            self.assertEqual([p.name for p in root_only], ['20mm_cube.stl'])
            self.assertEqual([p.name for p in recursive], ['20mm_cube.stl', 'alpha.stl'])

    def test_discover_stls_accepts_explicit_files_globs_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / 'nested'
            nested.mkdir()
            alpha = root / 'alpha.stl'
            beta = nested / 'beta.stl'
            alpha.write_text('solid alpha\nendsolid alpha\n')
            beta.write_text('solid beta\nendsolid beta\n')

            explicit = benchmark.discover_stls(
                root,
                stl_files=f'{alpha},{nested / "*.stl"}',
                max_stls=1,
            )

            self.assertEqual([p.name for p in explicit], ['alpha.stl'])

    def test_build_sweep_specs_cartesian_product(self):
        args = SimpleNamespace(
            adaptive_grid_resolutions=False,
            grid_resolutions='24,32',
            domain_scales='2.0',
            freestream_speeds='60,80',
            reynolds_numbers='5e4,1e5',
            step_counts=None,
            grid_resolution=32,
            domain_scale=2.0,
            freestream_speed=80.0,
            reynolds_number=1e5,
            steps=200,
            max_combinations=None,
        )

        sweep = benchmark.build_sweep_specs(args)

        self.assertEqual(len(sweep['combinations']), 8)
        self.assertEqual(sweep['combinations'][0]['grid_resolution'], 24)
        self.assertEqual(sweep['combinations'][0]['freestream_speed'], 60.0)
        self.assertEqual(sweep['combinations'][0]['reynolds_number'], 50000.0)
        self.assertEqual(sweep['axes']['grid_resolutions'], [24, 32])

    def test_adaptive_grid_resolution_scales_with_mesh_complexity(self):
        class Mesh:
            def __init__(self, face_count, extents, watertight):
                self.faces = [(0, 1, 2)] * face_count
                self.vertices = [(0.0, 0.0, 0.0)]
                self.extents = benchmark.np.asarray(extents, dtype=float)
                self.is_watertight = watertight

        simple = Mesh(686, [15.0, 15.0, 10.0], True)
        medium = Mesh(1_674, [236.0, 84.0, 340.0], True)
        complex_open = Mesh(37_298, [171.0, 189.0, 59.0], False)

        self.assertEqual(benchmark.estimate_adaptive_grid_resolutions(simple), [24])
        self.assertEqual(benchmark.estimate_adaptive_grid_resolutions(medium), [40])
        self.assertEqual(benchmark.estimate_adaptive_grid_resolutions(complex_open), [48])

    def test_make_case_uses_grid_resolution_for_block_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stl_path = root / 'geom.stl'
            stl_path.write_text(
                '\n'.join([
                    'solid geom',
                    'facet normal 0 0 1',
                    '  outer loop',
                    '    vertex 0 0 0',
                    '    vertex 1 0 0',
                    '    vertex 0 1 0',
                    '  endloop',
                    'endfacet',
                    'endsolid geom',
                ])
            )
            domain_min = benchmark.np.array([0.0, 0.0, 0.0], dtype=float)
            domain_max = benchmark.np.array([2.0, 2.0, 2.0], dtype=float)

            case = benchmark.make_case(
                stl_path,
                patch_name='geom',
                grid_resolution=18,
                domain_min=domain_min,
                domain_max=domain_max,
                freestream_speed=80.0,
                reynolds_number=1e5,
            )
            try:
                block_mesh = (case / 'system' / 'blockMeshDict').read_text()
                forces = (case / 'system' / 'forces').read_text()
                self.assertIn('(18 18 18)', block_mesh)
                self.assertIn('libs ("libforces.so")', forces)
                self.assertNotIn('functionObjectLibs', forces)
            finally:
                shutil.rmtree(case, ignore_errors=True)

    def test_cube_stl_text_is_watertight(self):
        try:
            import trimesh
        except Exception:
            self.skipTest('trimesh is not available')

        stl_text = benchmark._cube_stl_text(center=(0.0, 0.0, 0.0), edge_length=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            stl_path = Path(tmp) / 'cube.stl'
            stl_path.write_text(stl_text)
            mesh = trimesh.load_mesh(str(stl_path), force='mesh')
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.dump()))
            self.assertTrue(mesh.is_watertight)
            self.assertTrue(mesh.is_winding_consistent)
            self.assertEqual(len(mesh.faces), 12)

    def test_parse_openfoam_force_dat_vector_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            force_dat = Path(tmp) / 'force.dat'
            force_dat.write_text(
                '# Time forces(pressure viscous porous) moment(pressure viscous porous)\n'
                '0.1 (1 2 3) (4 5 6) (7 8 9) (0.1 0.2 0.3) (0.4 0.5 0.6) (0.7 0.8 0.9)\n'
            )

            parsed = benchmark._parse_forces_dat(
                force_dat,
                reference_area=2.0,
                density=1.0,
                freestream_speed=10.0,
            )

            self.assertEqual(parsed['force_x'], 12.0)
            self.assertEqual(parsed['force_y'], 15.0)
            self.assertEqual(parsed['force_z'], 18.0)
            self.assertAlmostEqual(parsed['moment_z'], 1.8)

    def test_write_force_dat_artifacts_creates_singular_and_plural_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)

            artifacts = benchmark._write_force_dat_artifacts(case, {
                'time_dir': '5e-05',
                'force_x': -10.0,
                'force_y': 1.0,
                'force_z': 2.0,
                'moment_x': 0.0,
                'moment_y': 0.0,
                'moment_z': 0.0,
            })

            self.assertTrue((case / 'postProcessing' / 'forces' / '5e-05' / 'force.dat').exists())
            self.assertTrue((case / 'postProcessing' / 'forces' / '5e-05' / 'forces.dat').exists())
            self.assertIn('force.dat', artifacts)

    def test_export_force_dat_artifacts_copies_report_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = root / 'case'
            force_dir = case / 'postProcessing' / 'forces' / '5e-05'
            force_dir.mkdir(parents=True)
            (force_dir / 'force.dat').write_text('0 -1 0 0 0 0 0\n')
            (force_dir / 'forces.dat').write_text('0 -1 0 0 0 0 0\n')
            results = {
                'cases': [{
                    'stl_path': str(REPO / 'F-18_Hornet.stl'),
                    'sweep_results': [{
                        'stl_path': str(REPO / 'F-18_Hornet.stl'),
                        'case_dir': str(case),
                        'grid_resolution': 32,
                        'openfoam': {
                            'force': {
                                'force_dat': 'postProcessing/forces/5e-05/force.dat',
                                'forces_dat': 'postProcessing/forces/5e-05/forces.dat',
                            },
                        },
                    }],
                }],
            }

            benchmark.export_force_dat_artifacts(results, root / 'out')

            self.assertTrue((root / 'out' / 'F_18_Hornet_grid32_force.dat').exists())
            self.assertTrue((root / 'out' / 'F_18_Hornet_grid32_forces.dat').exists())
            force_info = results['cases'][0]['sweep_results'][0]['openfoam']['force']
            self.assertIn('exported_force_dat', force_info)

    def test_build_timing_report_includes_solver_and_openfoam_totals(self):
        results = {
            'benchmark_root': str(REPO),
            'stl_count': 1,
            'benchmark_total_seconds': 13.0,
            'cases': [{
                'stl_path': str(REPO / 'F-18_Hornet.stl'),
                'sweep_results': [{
                    'stl_path': str(REPO / 'F-18_Hornet.stl'),
                    'grid_resolution': 32,
                    'steps': 200,
                    'error_percentage': 0.25,
                    'timings': {
                        'internal_solver_total_seconds': 2.0,
                        'openfoam_total_seconds': 10.0,
                        'openfoam_to_internal_speed_ratio': 5.0,
                    },
                    'openfoam': {
                        'status': 'completed',
                        'force': {
                            'source': 'postProcessing/forces.dat',
                        },
                        'commands': {
                            'sonicFoam': {
                                'returncode': 0,
                                'duration_seconds': 6.5,
                            },
                        },
                    },
                }],
            }],
        }

        report = benchmark.build_timing_report(results)

        self.assertIn('F-18_Hornet.stl', report)
        self.assertIn('2.000s', report)
        self.assertIn('10.000s', report)
        self.assertIn('5.00x', report)
        self.assertIn('postProcessing/forces.dat', report)
        self.assertIn('sonicFoam', report)

    def test_summarize_sweep_results_adds_sanity_gate_metadata(self):
        summary = benchmark.summarize_sweep_results([{
            'error_percentage': 0.25,
            'internal': {
                'drag_coefficient': 0.31,
            },
            'openfoam': {
                'status': 'completed',
                'force': {
                    'cd_total': 0.33,
                },
            },
        }])

        gate = summary['benchmark_gate']
        self.assertEqual(gate['status'], 'pass')
        self.assertEqual(gate['achieved_evidence_level'], 'solver_validation')
        self.assertEqual(gate['claim_scope'], 'sanity_only')
        self.assertTrue(gate['supports_sanity_claim'])
        self.assertFalse(gate['supports_claim_upgrade'])
        self.assertIn('Publication-quality validation', gate['blocked_claims'])
        self.assertIn('sanity', gate['summary'].lower())

    def test_build_timing_report_includes_benchmark_gate_section(self):
        results = {
            'benchmark_root': str(REPO),
            'stl_count': 1,
            'benchmark_total_seconds': 13.0,
            'cases': [{
                'stl_path': str(REPO / '20mm_cube.stl'),
                'sweep_results': [{
                    'stl_path': str(REPO / '20mm_cube.stl'),
                    'grid_resolution': 24,
                    'steps': 200,
                    'error_percentage': 0.5,
                    'timings': {
                        'internal_solver_total_seconds': 1.0,
                        'openfoam_total_seconds': 2.0,
                        'openfoam_to_internal_speed_ratio': 2.0,
                    },
                    'openfoam': {
                        'status': 'completed',
                        'force': {
                            'source': 'postProcessing/forces.dat',
                        },
                        'commands': {},
                    },
                }],
            }],
        }

        report = benchmark.build_timing_report(results)

        self.assertIn('## Benchmark Gate', report)
        self.assertIn('Status: PASS', report)
        self.assertIn('Claim scope: sanity_only', report)
        self.assertIn('Supports claim upgrade: no', report)
        self.assertIn('Publication-quality validation', report)

    def test_main_prints_top_level_benchmark_gate_metadata(self):
        stl_path = REPO / '20mm_cube.stl'
        fake_result = {
            'stl_path': str(stl_path),
            'sweep_results': [{
                'stl_path': str(stl_path),
                'grid_resolution': 24,
                'steps': 200,
                'error_percentage': 0.5,
                'surrogate_label_quality': 'high_accuracy',
                'openfoam': {
                    'status': 'completed',
                    'force': {
                        'cd_total': 0.2,
                    },
                },
            }],
            'summary': {},
        }

        with patch.object(benchmark, 'discover_stls', return_value=[stl_path]), patch.object(
            benchmark,
            'run_benchmark_for_stl',
            return_value=fake_result,
        ), patch.object(benchmark, '_is_windows_host', return_value=False):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = benchmark.main(['--stl-dir', str(REPO)])

        payload = json.loads(stdout.getvalue())
        gate = payload['benchmark_gate']
        self.assertEqual(exit_code, 0)
        self.assertEqual(gate['status'], 'pass')
        self.assertEqual(gate['claim_scope'], 'sanity_only')
        self.assertTrue(gate['supports_sanity_claim'])
        self.assertFalse(gate['supports_claim_upgrade'])


if __name__ == '__main__':
    unittest.main()
