# Solver Timing Report

Benchmark root: `D:\CodeProjects\research-paper`
STL count: 1
Total benchmark wall time: 51.194s

## Case Summary

| STL | Grid | Steps | Internal Solver Total | OpenFOAM Total | OpenFOAM/Internal | Error | Force Source | OpenFOAM Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| F-18_Hornet.stl | 32 | 200 | 4.572s | 46.051s | 10.07x | 0.000003% | manual_pressure_integration | completed |

## Aggregate Timing

Mean internal solver time: 4.572s
Mean OpenFOAM time: 46.051s
Mean OpenFOAM/internal ratio: 10.07x

## OpenFOAM Command Breakdown

### F-18_Hornet.stl grid 32

| Command | Return Code | Duration |
| --- | ---: | ---: |
| blockMesh | 0 | 0.684s |
| surfaceFeatureExtract | 0 | 0.313s |
| snappyHexMesh | 0 | 5.919s |
| checkMesh | 0 | 1.123s |
| sonicFoam | 0 | 34.293s |
| forces | 0 | 0.516s |
