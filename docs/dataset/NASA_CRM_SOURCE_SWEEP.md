# NASA CRM Source Sweep

- Generated: `2026-06-05`
- Catalog: `D:\CodeProjects\research-paper\docs\dataset\nasa_crm_source_candidates.json`
- Scope: public-source sweep for NASA Common Research Model (CRM) and adjacent aircraft-geometry / benchmark sources intended for future corpus intake.
- Boundary: this is a source-discovery artifact only. It does not modify builders, tests, or manifests, and it does not claim ingestion success.

## Method

- Prioritized official NASA CRM pages first, then adjacent public benchmark sources with strong geometry utility.
- Preferred primary sources and official mirrors.
- Recorded geometry scope explicitly as `whole_aircraft`, `semispan`, `component`, `airfoil_only`, or `validation_only`.
- Left licenses and flight-test claims blank when not stated on the source page.

## High-Value Official NASA CRM Families

| Family | Kind | Apparent records | Best source page | Notes |
|---|---|---:|---|---|
| CRM-HL reference geometry parts | component | 81 | https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/reference-geometry/ | Largest official CRM-HL part library; STEP zips for wing, body, flap, slat, nacelle, tail, gear, fairings, axes, positions. |
| CRM-HL assembled geometry | semispan | 2 current half-aircraft configs | https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/assembled-geometry/ | Half-aircraft solids for landing and takeoff; watertight but page warns small gaps should be sealed for CFD use. |
| CRM-HL NTF model geometry | whole_aircraft / semispan | 4 downloads | https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/model-specific-geometry/ | Explicit 2.7% full-span and 5.2% semispan wind-tunnel-model geometry. |
| CRM-HL DLR geometry | component | 1 | https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/dlr-geometry/ | DLR nacelle, pylon, and chine variant. |
| CRM-NLF geometry | whole_aircraft + airfoil pack | 3 | https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/crm-nlf/crm-nlf-geometry/ | Natural-laminar-flow CRM variant in IGES, STEP, and airfoil DAT package. |
| CRM icing geometry | whole_aircraft + semispan | 4 | https://commonresearchmodel.larc.nasa.gov/home-2/icing-research/ice-accretion-database-2/geometry-information/ | CRM65 full-aircraft CAD plus inboard, midspan, and outboard hybrid model assemblies. |
| DPW6 geometry release | whole_aircraft | 5 | https://commonresearchmodel.larc.nasa.gov/geometry/dpw6-geometries/ | STEP, IGES, CATIA, Parasolid, and change notes. |
| DPW7 geometry release | whole_aircraft | 8 | https://commonresearchmodel.larc.nasa.gov/geometry/dpw-7-geometries/ | WBT geometry variants across units / quality levels; mostly IGES and CATIA. |
| Legacy DPW4 / original CAD | whole_aircraft | 7 | https://commonresearchmodel.larc.nasa.gov/geometry/iges-files/ ; https://commonresearchmodel.larc.nasa.gov/geometry/stp-files/ ; https://commonresearchmodel.larc.nasa.gov/geometry/original-cad-files/ | Good provenance anchors for older CRM geometry states. |
| CRM65 airfoil sections | airfoil_only | 6 | https://commonresearchmodel.larc.nasa.gov/crm-65-airfoil-sections/ | Blunt / sharp trailing-edge section files, plots, and readme bundle. |
| Vertical-tail / upper-swept-strut extras | component | 3 | https://commonresearchmodel.larc.nasa.gov/geometry/vertical-tail-geometry/ ; https://commonresearchmodel.larc.nasa.gov/geometry/upper-swept-strut-geometry/ | Useful adjacent CRM-derived component studies. |
| Computational result CSVs | validation_only | 19 visible CSVs | https://commonresearchmodel.larc.nasa.gov/computational-results/ | Not geometry, but useful for pairing geometry variants with benchmark outputs. |

## Adjacent Public Benchmark Families

### NASA TMR and NASA-hosted benchmark geometry

- NASA Turbulence Modeling Resource landing page now states the authoritative site moved to the GitHub mirror and that old page URLs map directly by replacing `turbmodels.larc.nasa.gov/` with `tmbwg.github.io/turbmodels/`:
  - https://www.nasa.gov/nasa-turbulence-modeling-resource/
  - https://tmbwg.github.io/turbmodels/
- High-value geometry families discovered there:
  - Juncture Flow, turbulent F6-based geometry:
    - page: https://tmbwg.github.io/turbmodels/Other_exp_Data/JunctureFlow/junctureflow_geometry.html
    - exposes official NASA download links for Parasolid, STEP, IGES, cleaned STEP, in-tunnel assembly, and scan-derived STL / DAT support files.
    - geometry scope: whole-aircraft-like wing-body benchmark plus validation support.
  - Juncture Flow, turbulent symmetric wing geometry:
    - page: https://tmbwg.github.io/turbmodels/Other_exp_Data/JunctureFlow_symm/junctureflow_symm_geometry.html
    - exposes official NASA download links for Parasolid, STEP, IGES, in-tunnel assembly, and scan data.
    - geometry scope: whole-aircraft-like symmetric wing-body benchmark.
  - 2D multielement CRM-HL cut:
    - page: https://tmbwg.github.io/turbmodels/multielementverif_grids.html
    - direct geometry files: `crmhl-2dcut.igs`, `crmhl-2dcut.stp`, `crmhl-2dcut.egads`.
    - one NASA zip aggregates the geometry and grid families.
    - geometry scope: airfoil_only / high-lift section benchmark derived from CRM-HL.
  - NACA 0012 validation grids:
    - page: https://tmbwg.github.io/turbmodels/naca0012_grids.html
    - one NASA zip aggregates 2D PLOT3D, 3D PLOT3D, NMF, structured CGNS, and unstructured CGNS families.
    - geometry scope: airfoil_only / validation-heavy benchmark.

### NASA Glenn NPARC validation pages

- ONERA M6 wing:
  - case page: https://www.grc.nasa.gov/WWW/wind/valid/m6wing/m6wing.html
  - study archive: https://www.grc.nasa.gov/WWW/wind/valid/m6wing/m6wing01/m6wing01.html
  - geometry scope: semispan wing benchmark with airfoil coordinates, modified coordinates, scanned report pages, pressure traces, Tecplot support, and a study tarball with grid / run assets.
- RAE 2822:
  - page: https://www.grc.nasa.gov/www/wind/valid/raetaf/raetaf.html
  - direct geometry file: `geom.txt`
  - geometry scope: airfoil_only.
- NLR airfoil with flap:
  - page: https://www.grc.nasa.gov/www/wind/valid/nlrflap/nlrflap01/nlrflap01.html
  - direct archive: `nlrflap01.tar.Z`
  - geometry scope: airfoil_only / validation-heavy overset-grid case.

### Official NASA whole-aircraft adjacent family

- Urban Air Mobility reference vehicles:
  - page: https://www.nasa.gov/reference/uam-refs/
  - public OpenVSP models are posted for Tiltduct, Tiltwing, Multi-Tiltrotor, Quadrotor, Lift+Cruise, Side-by-Side, and Quiet Single Main Rotor.
  - geometry scope: whole_aircraft.
  - these are attractive because they add non-CRM aircraft families while staying on official NASA-hosted downloads.

### Large airfoil-only expansion lane

- UIUC Airfoil Coordinates Database:
  - page: https://m-selig.ae.illinois.edu/ads/coord_database.html
  - archive: https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip
  - page states approximately `1,650` airfoils in Version 2.0.
  - geometry scope: airfoil_only.
  - not an official NASA source, but it is public, citable, and immediately useful if the intake lane broadens beyond aircraft CAD.

## Acquisition Triage

### Tier 1: ingest next

- CRM-HL reference geometry page: `81` discrete STEP-part records, all official, directly downloadable, and already broken into machine-manageable files.
- CRM-HL assembled geometry: `2` half-aircraft solids with clear semantics and low discovery overhead.
- CRM-HL NTF model geometry: `2.7%` full-span bundle plus `5.2%` semispan files; highly useful because scale factors are explicit.
- CRM icing geometry: `4` files covering full-aircraft plus semispan hybrid models.
- CRM-NLF geometry: direct STEP / IGES / DAT variant family.
- DPW6 STEP release: easy benchmark bridge into the DPW ecosystem.

### Tier 2: expand benchmark coverage

- DPW7 geometry family: strong benchmark value, but conversion work is needed because the visible assets are IGES / CATIA-heavy.
- TMR Juncture Flow F6-based and symmetric-wing families: high validation value and official NASA download links, but some records mix CAD with scans and in-tunnel context.
- ONERA M6 and RAE 2822 NASA Glenn pages: inexpensive airfoil / semispan benchmark additions.
- NLR flap archive: useful if the intake scope wants multielement airfoil validation cases.

### Tier 3: whole-aircraft diversification

- NASA UAM reference OpenVSP models: at least `7` official whole-aircraft records across eVTOL configurations.
- These should be a priority once the CRM core is exhausted, because they reduce source-family leakage and broaden geometric diversity.

### Tier 4: large-volume airfoil lane

- UIUC airfoil archive alone can add roughly `1,650` airfoil-only records.
- This should be gated behind an explicit airfoil-only intake policy because it changes the corpus balance substantially and is not comparable to full-aircraft CAD.

## Blockers And Handling Notes

- Many NASA CRM pages do not state an explicit per-file license. Treat them as public research downloads, preserve page provenance, and do not invent a license string.
- DPW7, IGES pages, CATIA packages, Parasolid packages, and some original CAD releases require a format-conversion lane if the downstream intake standard stays STEP-first.
- The CRM-HL assembled page explicitly warns its watertight solids may still contain gaps relevant to CFD grid generation.
- TMR discovery now depends on the GitHub-hosted mirror for page structure, even when the direct downloadable payload is still hosted on `nasa.gov`.
- Validation-only CSVs and pressure traces are valuable linkage data, but they should not be confused with geometry records.

## Path To A Few Hundred Records

1. Enumerate the official CRM-HL reference-geometry page into all `81` part zips and add the current assembled / model-specific / icing / CRM-NLF / DPW6 files. That alone yields roughly `100+` CRM-family geometry candidates.
2. Add DPW7, original CAD, CRM65 airfoil, vertical-tail, and upper-swept-strut families. That pushes the CRM ecosystem well beyond `120` public candidates.
3. Add NASA UAM OpenVSP vehicles and TMR Juncture Flow geometry families. That raises whole-aircraft and benchmark diversity while staying mostly inside NASA-hosted downloads.
4. Add ONERA M6, RAE 2822, and NLR flap benchmark files from NASA Glenn validation pages for lower-cost validation geometry coverage.
5. If an airfoil-only lane is approved, ingest the UIUC archive last. That is the simplest route from `~150` candidates to well over `1,500`.

## Recommended First Pass

- First pass target: all `ready` official NASA CRM geometry candidates, plus NASA UAM OpenVSP models.
- Second pass target: `format_conversion_needed` CRM and TMR families.
- Holdout / metadata-only lane: validation CSVs, pressure traces, scanned geometry-reference PDFs, and pages whose only payload is documentation.
