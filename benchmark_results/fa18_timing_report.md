# Solver Timing Report

Benchmark root: `D:\CodeProjects\research-paper`
STL count: 1
Total benchmark wall time: 53.870s

## Case Summary

| STL | Grid | Steps | Internal Solver Total | OpenFOAM Total | OpenFOAM/Internal | Error | Force Source | OpenFOAM Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| F-18_Hornet.stl | 32 | 200 | 4.976s | 48.244s | 9.69x | 0.000003% | manual_pressure_integration_forces_dat | completed |

## Aggregate Timing

Mean internal solver time: 4.976s
Mean OpenFOAM time: 48.244s
Mean OpenFOAM/internal ratio: 9.69x

## OpenFOAM Command Breakdown

### F-18_Hornet.stl grid 32

| Command | Return Code | Duration |
| --- | ---: | ---: |
| blockMesh | 0 | 0.647s |
| surfaceFeatureExtract | 0 | 0.273s |
| snappyHexMesh | 0 | 5.887s |
| checkMesh | 0 | 1.200s |
| sonicFoam | 0 | 35.679s |
| forces | 0 | 0.550s |
