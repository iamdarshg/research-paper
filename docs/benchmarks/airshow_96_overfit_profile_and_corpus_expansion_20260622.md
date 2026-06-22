# 96 Cubed Overfit, Profiling, And Corpus Expansion Report

Generated: `2026-06-22`

## Scope

This report records the follow-up requested after the `96^3` generated artifacts looked like dense blobs/slabs rather than aircraft. The work tightened the aircraft-validity gate, profiled a continuation epoch, tested for overfit/drift, built a `96^3` NASA CRM add-on corpus, added per-record flight-path profiles, and ran an expanded-corpus continuation.

## Source Grounding

- NASA CRM home page: https://commonresearchmodel.larc.nasa.gov/
- NASA CRM-HL reference geometry page: https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/reference-geometry/
- NASA CRM-HL model-specific geometry page: https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/model-specific-geometry/
- NASA high-speed CRM STP files page: https://commonresearchmodel.larc.nasa.gov/geometry/stp-files/
- Local source catalog: `docs/dataset/nasa_crm_source_catalog.json`

The NASA CRM pages describe public CRM geometry and STEP/CAD downloads. The local builder uses the checked-in catalog entries and records the original source URLs, hashes, converted STL paths, voxel paths, and bounded design-spec provenance.

## Validity Gate Changes

The old gate let dense symmetric artifacts pass because symmetry, span, length, and longitudinal variation were enough. Two stricter checks were added:

- `planform_sparsity`: rejects filled top-view planforms and dense occupied bounding boxes.
- `fuselage_end_presence`: requires centerline occupancy near both longitudinal ends and enough centerline coverage.

The occupancy floor was reduced from `0.005` to `0.002` after rebuilding NASA CRM at `96^3`: real thin transport CAD occupied only about `0.0038-0.0044` of the lattice, while still passing the stricter planform and fuselage checks.

Evidence:

- Tests: `python -m pytest tests\test_aircraft_validity.py -q` -> `8 passed`
- Strict rescore of prior generated artifacts: `build/airshow_direct_solver_extended_20260621/g96_ext_lr1e5_ep3/flight_path_sym_topk05/aircraft_validity_strict_rescore.json`
- Result: prior generated outputs now fail `0/3` pass, mainly on `planform_sparsity` and `fuselage_end_presence`.

## Profiled Continuation

Starting checkpoint:

`build/airshow_direct_solver_extended_20260621/g96_ext_lr1e5_ep3/checkpoints/final_optimized_model.pt`

Profiled continuation output:

`build/airshow_direct_solver_extended_20260622/g96_ext_lr5e6_ep4/checkpoints/final_optimized_model.pt`

Profiler artifact:

`build/airshow_profile_20260622/g96_ep4_pyspy.svg`

Top visible flamegraph regions:

- `advanced_lbm_solver.py::collide_and_stream`, especially `_apply_bfl_boundary`
- `sdf_utils.py::compute_all_link_distances`
- coordinate decoder forward paths in `aircraft_diffusion_cfd.py`
- Adam optimizer step overhead

This supports the earlier suspicion that the run is not cleanly compute-saturating the GPU. The solver path repeatedly pays Python/SDF/boundary work around sparse direct-solver evaluations, while the GPU has high memory footprint and intermittent utilization.

## Overfit/Drift Check

| Run | Manifest | Optimizer loss | Direct solver eval loss | Strict generated validity |
| --- | --- | ---: | ---: | --- |
| Previous best `ep3` | Airshow `96^3` | `0.399479` | `0.972279` | `0/3` after stricter rescore |
| Profiled `ep4` | Airshow `96^3` | `0.383894` | `1.466760` | `0/3` |

The optimizer scalar improved, but the scheduled direct-solver evaluation worsened. That is an overfit/drift signal, not progress toward aircraft generation.

Generated `ep4` artifacts:

`build/airshow_direct_solver_extended_20260622/g96_ext_lr5e6_ep4/flight_path_sym_topk05/flight_path_results.json`

All three generated flight-path cases failed the strict aircraft gate.

## NASA CRM 96 Cubed Add-On

Built with:

`python CLI\build_nasa_crm_whole_aircraft_context.py --output-root build\nasa_crm_whole_aircraft_g96 --manifest build\nasa_crm_whole_aircraft_g96\manifest.jsonl --provenance build\nasa_crm_whole_aircraft_g96\provenance.json --report docs\dataset\nasa_crm_whole_aircraft_g96_20260622.md --grid-size 96 --simulation-steps 5 --analysis-device cuda --skip-refinement`

Results:

- Selected records: `15`
- Built records: `15`
- Errors: `0`
- Basic manifest validation: pass
- Aircraft validity: `15/15` pass at `96^3`

Artifacts:

- Manifest: `build/nasa_crm_whole_aircraft_g96/manifest.jsonl`
- Provenance: `build/nasa_crm_whole_aircraft_g96/provenance.json`
- Validity: `build/nasa_crm_whole_aircraft_g96/aircraft_validity.json`
- Report: `docs/dataset/nasa_crm_whole_aircraft_g96_20260622.md`

## Expanded Flight-Path Manifest

The Airshow `96^3` corpus and NASA CRM `96^3` add-on were packed into one manifest with deterministic per-record flight-path profiles.

Artifacts:

- Combined manifest: `build/expanded_aircraft_corpus_20260622/manifest.jsonl`
- Flight-path report: `build/expanded_aircraft_corpus_20260622/flight_path_manifest_report.json`
- Manifest validation: `build/expanded_aircraft_corpus_20260622/manifest_validation.json`

Summary:

- Total records: `370`
- Airshow records: `355`
- NASA CRM records: `15`
- Grid shapes: `[96, 96, 96]` for all `370`
- Missing geometry: `0`

Boundary: flight paths are deterministic conditioning profiles derived from manifest design-spec fields. They are not measured flight trajectories.

## Expanded-Corpus Training

Run:

`build/expanded_aircraft_training_20260622/g96_airshow_nasa_lr5e6_ep1/checkpoints/final_optimized_model.pt`

Metrics:

- Optimizer loss: `7.100261`
- MSE: `5.317817`
- Geometry reconstruction: `0.084054`
- Generation reconstruction: `0.038669`
- Consistency: `1.650262`
- Direct solver eval loss: `1.458189`
- Direct solver eval count: `12`

Resource summary:

- Elapsed: `246.313 s`
- Peak GPU memory used: `7945 MB`
- Mean GPU utilization: `42.85%`
- Peak GPU utilization: `100%`
- Mean process CPU: `318.84%`
- Peak process RSS: `6231 MB`

Generated outputs:

`build/expanded_aircraft_training_20260622/g96_airshow_nasa_lr5e6_ep1/flight_path_sym_topk05/flight_path_results.json`

All three generated flight-path cases failed the strict aircraft gate:

- `short_takeoff_payload`: `planform_sparsity`, `fuselage_end_presence`
- `high_speed_sprint`: `body_centerline_dominance`, `planform_sparsity`, `fuselage_end_presence`
- `endurance_turning`: `body_centerline_dominance`, `planform_sparsity`, `fuselage_end_presence`

## Conclusion

Objectively, this does not yet make actual aircraft. The stricter gate correctly rejects the generated artifacts and accepts the NASA CRM `96^3` public geometry. The Airshow-only continuation shows overfit/drift, and the one-epoch expanded Airshow+NASA continuation destabilizes reconstruction without improving generated validity.

The next technically credible route is not more blind epochs on the same architecture. It is a curriculum or representation change that gives sparse transport geometry much more influence: train or fine-tune on source-valid whole-aircraft records first, weight NASA/known-aircraft records higher, keep the direct solver loss, and treat strict aircraft validity as the stop condition.
