import tempfile
import unittest
import shutil
from pathlib import Path
from types import SimpleNamespace
import sys


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


if __name__ == '__main__':
    unittest.main()
