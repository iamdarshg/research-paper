
import os
import tempfile
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
from skimage import measure
import trimesh
from advanced_lbm_solver import D3Q27CascadedSolver
from config import CFDConfig, LBMPhysicsConfig, OPENFOAM_AVAILABLE, OPENFOAM_ROOT, MissionProfile
from constraints import ConstraintProjector, ConstraintReport
from geometry import TypedAircraftGeometry, AircraftPart

class AdvancedCFDSimulator:
    """Advanced CFD simulator with FluidX3D integration and adaptive mesh refinement"""

    def __init__(self, config: CFDConfig, device: torch.device):
        self.config = config
        self.device = device
        self.resolution = config.base_grid_resolution

        self.lbm_solver = D3Q27CascadedSolver(self.config, device, LBMPhysicsConfig)

        if self.config.use_amr:
            import copy
            amr_config = copy.deepcopy(self.config)
            amr_config.resolution = self.config.base_grid_resolution * 2
            self.amr_solver = D3Q27CascadedSolver(amr_config, device, LBMPhysicsConfig)
        else:
            self.amr_solver = None
        self.init_flow_field()

    def set_resolution(self, resolution: int):
        if resolution == self.resolution:
            return
        self.resolution = resolution
        self.config.base_grid_resolution = resolution
        self.lbm_solver = D3Q27CascadedSolver(self.config, self.device, LBMPhysicsConfig)
        if self.config.use_amr:
            import copy
            amr_config = copy.deepcopy(self.config)
            amr_config.resolution = resolution * 2
            self.amr_solver = D3Q27CascadedSolver(amr_config, self.device, LBMPhysicsConfig)
        else:
            self.amr_solver = None
        self.init_flow_field()
        print(f"AdvancedCFDSimulator: Solver resolution updated to {resolution}^3")

    def init_flow_field(self):
        self.lbm_solver._initialize_equilibrium()
        if self.amr_solver:
            self.amr_solver._initialize_equilibrium()

    def simulate_aerodynamics(self, geometry: Union[torch.Tensor, TypedAircraftGeometry], steps: int = 100,
                               mission: Optional[MissionProfile] = None,
                               existing_report: Optional[ConstraintReport] = None) -> Dict[str, Any]:
        # Handle TypedAircraftGeometry (Issue #16)
        if isinstance(geometry, TypedAircraftGeometry):
            typed_geom = geometry
            occupancy = typed_geom.get_combined_occupancy()
        else:
            occupancy = geometry
            typed_geom = None

        # Apply Constraint Projection if mission is provided and not already done
        if mission:
            projector = ConstraintProjector(self.resolution, device=self.device, existing_report=existing_report)
            if typed_geom is None:
                # If we only have raw geometry, treat as fuselage for legacy compatibility
                typed_geom = TypedAircraftGeometry(self.resolution, device=self.device)
                typed_geom.set_part_mask(AircraftPart.FUSELAGE, occupancy)

            typed_geom = projector.project(typed_geom, mission)
            occupancy = typed_geom.get_combined_occupancy()
            constraint_report = projector.get_report(typed_geom)
        else:
            constraint_report = existing_report.to_dict() if existing_report else {"valid": True, "repaired": False, "violations": []}

        geometry_mask = (occupancy > 0.5).float()
        self.lbm_solver.collide_stream(geometry_mask, steps=steps)
        results = self.lbm_solver.compute_aerodynamic_coefficients(geometry_mask)

        # Add physics feasibility to report (Issue #16)
        if mission and typed_geom:
            projector = ConstraintProjector(self.resolution, device=self.device, existing_report=existing_report)
            feasibility = projector.check_feasibility(typed_geom, results, mission)
            # Refresh report after feasibility checks
            constraint_report = projector.get_report(typed_geom)
            results['feasibility'] = feasibility

        results['constraints'] = constraint_report

        # Initial fields from internal LBM
        results['velocity_fields'] = (self.lbm_solver.velocity_x, self.lbm_solver.velocity_y, self.lbm_solver.velocity_z)
        results['pressure_field'] = self.lbm_solver.pressure
        results['pinn_ready'] = False # Default False, only True if external validated truth exists

        if self.amr_solver:
            amr_geometry = F.interpolate(
                occupancy.unsqueeze(0).unsqueeze(0),
                size=(self.amr_solver.resolution, self.amr_solver.resolution, self.amr_solver.resolution),
                mode='nearest'
            ).squeeze(0).squeeze(0)
            amr_geometry_mask = (amr_geometry > 0.5).float()
            self.amr_solver.collide_stream(amr_geometry_mask, steps=steps)
            amr_results = self.amr_solver.compute_aerodynamic_coefficients(amr_geometry_mask)
            results['drag_coefficient'] = (results['drag_coefficient'] + amr_results['drag_coefficient']) / 2
            results['lift_coefficient'] = (results['lift_coefficient'] + amr_results['lift_coefficient']) / 2

        # Issue #15: Multi-fidelity promotion
        external_results = None
        if mission and getattr(mission, 'force_external_validation', False):
            external_results = self._run_external_validation(occupancy, force=True)
        else:
            external_results = self._run_external_validation(occupancy)

        if external_results:
            # For high-fidelity ground truth, if independent PDE results exist,
            # they supersede the LBM surrogate entirely.
            results['external_ground_truth'] = external_results

            # Promote metadata to top level (Source promotion)
            results['drag_coefficient'] = external_results['drag_coefficient']
            results['lift_coefficient'] = external_results['lift_coefficient']
            results['physical_force_source'] = external_results.get('physical_force_source', results.get('physical_force_source'))
            results['label_source'] = external_results.get('label_source', 'External')
            results['label_tier'] = external_results.get('label_tier', 'external_pde')
            results['source'] = results['label_source']

            # Attach external PDE fields if available
            has_fields = False
            if 'velocity_fields' in external_results:
                results['velocity_fields'] = external_results['velocity_fields']
                has_fields = True
            if 'pressure_field' in external_results:
                results['pressure_field'] = external_results['pressure_field']

            # PINN-ready gate: Tier must be external AND fields must be verified
            results['pinn_ready'] = bool(
                external_results.get('pinn_ready', False) and
                results['label_tier'] == 'external_pde' and
                has_fields
            )
        return results

    def _run_external_validation(self, voxel_grid: torch.Tensor, force: bool = False) -> Optional[Dict[str, float]]:
        import random
        # Intelligent sampling: only run validation based on configured probability
        if not force and random.random() > self.config.validation_probability:
            return None

        if OPENFOAM_AVAILABLE:
            return self._run_openfoam_validation(voxel_grid)
        try:
            stl_path = self._voxel_to_stl_path(voxel_grid)
            if stl_path and os.path.exists(stl_path):
                return self._run_fluidx3d_fast(stl_path)
        except Exception as e:
            print(f"External validation failed: {e}")
        return None

    def _run_openfoam_validation(self, voxel_grid: torch.Tensor) -> Optional[Dict[str, float]]:
        from generator import OptimizedAircraftGenerator
        print("🚀 Running independent OpenFOAM validation for PINN ground truth...")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # We need an exporter instance, or a static method.
                # For now let's assume we can use basic logic or move export_openfoam_case to a util
                # Actually, export_openfoam_case was in AdvancedCFDSimulator in the original file
                # but I missed copying it here. Let me add it.
                case_info = self.export_openfoam_case(voxel_grid, tmp_dir)
                case_path = Path(case_info['case_dir'])
                from subprocess import run as sp_run

                if not OPENFOAM_AVAILABLE:
                    return None

                commands = [
                    "blockMesh",
                    "surfaceFeatureExtract",
                    "snappyHexMesh -overwrite",
                    "sonicFoam",
                    "postProcess -func sample -latestTime"
                ]
                for cmd in commands:
                    log_file = case_path / f"log.{cmd.split()[0]}"
                    proc = sp_run(f"bash -lc 'source {OPENFOAM_ROOT}/etc/bashrc && {cmd}' > {log_file} 2>&1", shell=True, cwd=case_path)
                    if proc.returncode != 0:
                        return None

                # Check for convergence in logs
                log_sonic = case_path / "log.sonicFoam"
                converged = False
                if log_sonic.exists():
                    log_content = log_sonic.read_text()
                    # Stricter convergence gate: residuals must be below 1e-4
                    import re
                    # Look for lines like "Final residual = 1.234e-05"
                    residuals = re.findall(r"Final residual = ([\d.e-]+)", log_content)
                    if residuals:
                        last_residuals = [float(r) for r in residuals[-5:]]
                        # All last residuals must be below threshold, and must have terminated correctly
                        if all(r < 1e-4 for r in last_residuals) and "End" in log_content:
                            converged = True
                    # Fail-closed: No fallback to simple "End" check to ensure PDE integrity

                force_file = case_path / "postProcessing" / "forces" / "0" / "force.dat"
                if force_file.exists():
                    with open(force_file, 'r') as f:
                        lines = [l for l in f.readlines() if not l.startswith('#')]
                        if lines:
                            last_line = lines[-1].split()
                            fx = float(last_line[1].replace('(', ''))
                            fz = float(last_line[3].replace(')', ''))
                            rho, U, ref_area = 1.225, 80.0, 1.0
                            dyn_pres = 0.5 * rho * U**2 * ref_area

                            of_results = {
                                'drag_coefficient': fx / dyn_pres,
                                'lift_coefficient': fz / dyn_pres,
                                'physical_force_source': fx,
                                'label_source': 'OpenFOAM',
                                'label_tier': 'external_pde',
                                'source': 'OpenFOAM',
                                'pinn_ready': False # Default False, only True if converged AND fields present
                            }

                            # Extract sampled fields
                            res = self.config.base_grid_resolution
                            sample_dir = case_path / "postProcessing" / "sample"

                            fields_present = False
                            if sample_dir.exists():
                                time_dirs = [d for d in sample_dir.iterdir() if d.is_dir()]
                                if time_dirs:
                                    # Find latest time dir in sample
                                    latest_sample_time = max(time_dirs, key=lambda d: float(d.name))

                                    # Use glob patterns for robust matching
                                    u_files = list(latest_sample_time.glob("*_U.xy"))
                                    p_files = list(latest_sample_time.glob("*_p.xy"))

                                    if u_files and p_files:
                                        u_file, p_file = u_files[0], p_files[0]
                                        u_data = np.loadtxt(u_file)
                                        p_data = np.loadtxt(p_file)

                                        if u_data.shape[0] == res**3 and p_data.shape[0] == res**3:
                                            # x, y, z, ux, uy, uz
                                            ux = torch.from_numpy(u_data[:, 3]).view(res, res, res).float().to(self.device)
                                            uy = torch.from_numpy(u_data[:, 4]).view(res, res, res).float().to(self.device)
                                            uz = torch.from_numpy(u_data[:, 5]).view(res, res, res).float().to(self.device)
                                            # x, y, z, p
                                            p_field = torch.from_numpy(p_data[:, 3]).view(res, res, res).float().to(self.device)

                                            of_results['velocity_fields'] = (ux, uy, uz)
                                            of_results['pressure_field'] = p_field
                                            fields_present = True
                                            print(f"✅ Successfully extracted OpenFOAM fields at {res}^3")

                            # pinn_ready requires strict convergence AND field presence
                            of_results['pinn_ready'] = converged and fields_present
                            return of_results
                return None
        except Exception as e:
            print(f"OpenFOAM validation failed: {e}")
            return None

    def _voxel_to_stl_path(self, voxel_grid: torch.Tensor) -> Optional[str]:
        try:
            voxel_np = voxel_grid.cpu().numpy()
            binary_grid = (voxel_np > 0.5).astype(np.float32)
            vertices, faces, _, _ = measure.marching_cubes(binary_grid, level=0.5, spacing=(1.0, 1.0, 1.0))
            scale = float(self.config.lbm_config.physical_length_scale)
            h = scale / float(self.config.base_grid_resolution)
            vertices = vertices * h - (scale * 0.5) + (0.5 * h)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                mesh.export(tmp.name)
                return tmp.name
        except Exception as e:
            print(f"STL conversion failed: {e}")
            return None

    def _run_fluidx3d_fast(self, stl_path: str) -> Optional[Dict[str, float]]:
        # Hard-coded fallbacks removed to ensure data integrity
        print("⚠️ FluidX3D placeholder reached. No ground truth returned.")
        return None

    def export_openfoam_case(self, voxel_grid: torch.Tensor, case_dir: str) -> Dict[str, Any]:
        # Implementation from original AdvancedCFDSimulator (moved to AdvancedCFDSimulator for consistency)
        # Re-using the same logic.
        from generator import OptimizedAircraftGenerator
        case_path = Path(case_dir)
        tri_surface = case_path / "constant" / "triSurface"
        system = case_path / "system"
        constant = case_path / "constant"
        for p in (tri_surface, system, constant / "polyMesh", case_path / "0"):
            p.mkdir(parents=True, exist_ok=True)

        processed = voxel_grid # Simple for now
        stl_path = tri_surface / "design.stl"
        # Need a simple way to write STL here without generator circular dependency if possible
        # but generator.voxels_to_stl is useful.
        # Let's just use trimesh directly here to avoid circular dependency.
        voxel_np = processed.cpu().numpy()
        binary_grid = (voxel_np > 0.5).astype(np.float32)
        vertices, faces, _, _ = measure.marching_cubes(binary_grid, level=0.5, spacing=(1.0, 1.0, 1.0))
        scale = float(self.config.lbm_config.physical_length_scale)
        h = scale / float(self.config.base_grid_resolution)
        vertices = vertices * h - (scale * 0.5) + (0.5 * h)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(str(stl_path))

        (system / "blockMeshDict").write_text("""FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object blockMeshDict;\n}\nconvertToMeters 1;\nvertices\n(\n    (-5 -2 -2)\n    ( 5 -2 -2)\n    ( 5  2 -2)\n    (-5  2 -2)\n    (-5 -2  2)\n    ( 5 -2  2)\n    ( 5  2  2)\n    (-5  2  2)\n);\nblocks\n(\n    hex (0 1 2 3 4 5 6 7) (60 24 24) simpleGrading (1 1 1)\n);\nedges ( );\nboundary\n(\n    inlet { type patch; faces ((0 4 7 3)); }\n    outlet { type patch; faces ((1 2 6 5)); }\n    top { type patch; faces ((3 7 6 2)); }\n    bottom { type patch; faces ((0 1 5 4)); }\n    front { type symmetryPlane; faces ((0 3 2 1)); }\n    back { type symmetryPlane; faces ((4 5 6 7)); }\n);\nmergePatchPairs ( );\n""")
        (system / "snappyHexMeshDict").write_text("""FoamFile\n{ version 2.0; format ascii; class dictionary; object snappyHexMeshDict; }\ncastellatedMesh true;\nsnap true;\naddLayers false;\nmergeTolerance 1e-6;\ngeometry\n{ design.stl { type triSurfaceMesh; name design; } }\ncastellatedMeshControls\n{\n    maxLocalCells 50000; maxGlobalCells 200000; minRefinementCells 0; nCellsBetweenLevels 2;\n    features ( ); refinementSurfaces { design { level (1 2); } }; refinementRegions { };\n    allowFreeStandingZoneFaces true; resolveFeatureAngle 30; locationInMesh (4 0 0);\n}\nsnapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; }\naddLayersControls { relativeSizes true; layers { } expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 30; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 0; }\nmeshQualityControls { maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }\n""")
        (system / "controlDict").write_text("""FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n    location    \"system\";\n    object      controlDict;\n}\napplication     sonicFoam;\nstartFrom       latestTime;\nstartTime       0;\nstopAt          endTime;\nendTime         0.0027;\ndeltaT          4e-08;\nwriteControl    runTime;\nwriteInterval   2e-04;\npurgeWrite      0;\nwriteFormat     ascii;\nwritePrecision  6;\nwriteCompression off;\ntimeFormat      general;\ntimePrecision   6;\nrunTimeModifiable true;\n""")
        (system / "fvSchemes").write_text("""FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n    location    \"system\";\n    object      fvSchemes;\n}\nddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }\ndivSchemes { default none; div(phi,U) Gauss limitedLinearV 1; div(phi,e) Gauss limitedLinear 1; div(phid,p) Gauss limitedLinear 1; div(phiv,p) Gauss limitedLinear 1; div(phi,K) Gauss limitedLinear 1; div(phi,k) Gauss upwind; div(phi,epsilon) Gauss upwind; div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear; }\nlaplacianSchemes { default Gauss linear limited corrected 0.5; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n""")
        (system / "fvSolution").write_text("""FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n    location    \"system\";\n    object      fvSolution;\n}\nsolvers { \"rho.*\" { solver diagonal; } \"p.*\" { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; } \"(U|e).*\" { $p; tolerance 1e-9; } \"(k|epsilon).*\" { $p; tolerance 1e-10; } }\nPIMPLE { nOuterCorrectors 1; nCorrectors 2; nNonOrthogonalCorrectors 0; }\n""")
        (constant / "thermophysicalProperties").write_text("""FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n    location    \"constant\";\n    object      thermophysicalProperties;\n}\nthermoType { type hePsiThermo; mixture pureMixture; transport const; thermo hConst; equationOfState perfectGas; specie specie; energy sensibleInternalEnergy; }\nmixture { specie { molWeight 28.9; } thermodynamics { Cp 1005; Hf 0; } transport { mu 0; Pr 0.7; } }\n""")
        (constant / "turbulenceProperties").write_text("""FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; } simulationType laminar;\n""")
        (system / "forces").write_text("""FoamFile\n{ version 2.0; format ascii; class dictionary; object forces; }\ntype forces;\nfunctionObjectLibs (\"libforces.so\");\npatches (design);\nrho rho;\nrhoInf 1.225;\np p;\nU U;\nCofR (0 0 0);\n""")

        # Add sampling for field extraction matching voxel grid
        res = self.config.base_grid_resolution
        points_str = ""
        for x in range(res):
            for y in range(res):
                for z in range(res):
                    # Map to centered unit cube [-0.5, 0.5]
                    xp = (x / (res - 1)) - 0.5
                    yp = (y / (res - 1)) - 0.5
                    zp = (z / (res - 1)) - 0.5
                    points_str += f"({xp} {yp} {zp})\n"

        (system / "sample").write_text(f"""FoamFile\n{{ version 2.0; format ascii; class dictionary; object sample; }}\ntype sets;\nlibs ("libsampling.so");\ninterpolationScheme cell;\nsetFormat raw;\nsets\n(\n    voxelGrid\n    {{\n        type cloud;\n        axis xyz;\n        points\n        (\n            {points_str}\n        );\n    }}\n);\nfields (U p);\n""")

        (case_path / "0" / "U").write_text("""FoamFile\n{ version 2.0; format ascii; class volVectorField; object U; }\ndimensions [0 1 -1 0 0 0 0];\ninternalField uniform (80 0 0);\nboundaryField { inlet { type fixedValue; value uniform (80 0 0); } outlet { type pressureInletOutletVelocity; value uniform (80 0 0); } top { type slip; } bottom { type slip; } front { type symmetryPlane; } back { type symmetryPlane; } design { type noSlip; } }\n""")
        (case_path / "0" / "p").write_text("""FoamFile\n{ version 2.0; format ascii; class volScalarField; object p; }\ndimensions [1 -1 -2 0 0 0 0];\ninternalField uniform 101325;\nboundaryField { inlet { type totalPressure; p0 uniform 101325; value uniform 101325; } outlet { type fixedValue; value uniform 101325; } top { type zeroGradient; } bottom { type zeroGradient; } front { type symmetryPlane; } back { type symmetryPlane; } design { type zeroGradient; } }\n""")
        (case_path / "0" / "T").write_text("""FoamFile\n{ version 2.0; format ascii; class volScalarField; object T; }\ndimensions [0 0 0 1 0 0 0];\ninternalField uniform 300;\nboundaryField { inlet { type fixedValue; value uniform 300; } outlet { type zeroGradient; } top { type zeroGradient; } bottom { type zeroGradient; } front { type symmetryPlane; } back { type symmetryPlane; } design { type zeroGradient; } }\n""")
        (case_path / "0" / "rho").write_text("""FoamFile\n{ version 2.0; format ascii; class volScalarField; object rho; }\ndimensions [1 -3 0 0 0 0 0];\ninternalField uniform 1.225;\nboundaryField { inlet { type fixedValue; value uniform 1.225; } outlet { type zeroGradient; } top { type zeroGradient; } bottom { type zeroGradient; } front { type symmetryPlane; } back { type symmetryPlane; } design { type zeroGradient; } }\n""")
        return {"case_dir": str(case_path), "stl_path": str(stl_path), "openfoam_available": OPENFOAM_AVAILABLE}
