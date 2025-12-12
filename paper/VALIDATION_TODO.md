# Validation and Data Harvesting TODO

This document outlines a comprehensive plan for the validation and data harvesting process for the generative design of aircraft structures.

## 1. Experimental Plan

### 1.1. Component Validation Experiments

*   **Experiment C1: Airfoil Validation**
    *   **Objective:** Validate the CFD solver against experimental data for a standard airfoil (e.g., NACA 2412).
    *   **Procedure:**
        1.  Obtain experimental data for the lift and drag coefficients of the airfoil at various angles of attack and Reynolds numbers.
        2.  Simulate the flow over the airfoil using the CFD solver.
        3.  Compare the simulated lift and drag coefficients to the experimental data.
    *   **Success Criteria:** The simulated results should be within 5% of the experimental data.

*   **Experiment C2: Wing Validation**
    *   **Objective:** Validate the CFD solver against experimental data for a simple wing geometry.
    *   **Procedure:**
        1.  Obtain experimental data for the lift and drag coefficients of the wing at various angles of attack and Reynolds numbers.
        2.  Simulate the flow over the wing using the CFD solver.
        3.  Compare the simulated lift and drag coefficients to the experimental data.
    *   **Success Criteria:** The simulated results should be within 10% of the experimental data.

### 1.2. System Validation Experiments

*   **Experiment S1: Baseline Aircraft Comparison**
    *   **Objective:** Compare the performance of the generated designs to that of a baseline aircraft (e.g., a Cessna 172).
    *   **Procedure:**
        1.  Generate a set of aircraft designs using the generative design framework.
        2.  Evaluate the aerodynamic performance of the generated designs using the CFD solver.
        3.  Compare the lift-to-drag ratio of the generated designs to that of the baseline aircraft.
    *   **Success Criteria:** At least 10% of the generated designs should have a higher lift-to-drag ratio than the baseline aircraft.

*   **Experiment S2: Wind Tunnel Testing**
    *   **Objective:** Validate the performance of a generated design using wind tunnel testing.
    *   **Procedure:**
        1.  Select the most promising design from Experiment S1.
        2.  3D print a model of the design.
        3.  Test the model in a wind tunnel to measure its lift and drag coefficients.
        4.  Compare the wind tunnel results to the CFD simulations.
    *   **Success Criteria:** The wind tunnel results should be within 15% of the CFD simulations.

## 2. Data Harvesting Plan

### 2.1. Metrics to Collect

*   **Aerodynamic Performance:**
    *   Lift coefficient ($C_L$)
    *   Drag coefficient ($C_D$)
    *   Lift-to-drag ratio ($L/D$)
    *   Pressure distribution
    *   Velocity field
*   **Structural Properties:**
    *   Volume
    *   Surface area
    *   Connectivity
*   **Computational Performance:**
    *   Training time
    *   Inference time
    *   Memory usage

### 2.2. Data Sources for Comparison

*   **Experimental Data:**
    *   NACA airfoil data
    *   University of Illinois at Urbana-Champaign Airfoil Data Site
    *   NASA's National Advisory Committee for Aeronautics (NACA) reports
*   **Real-World Aircraft Data:**
    *   Jane's All the World's Aircraft
    *   Manufacturer's specifications

## 3. High-Performance Targets

*   **Lift-to-Drag Ratio:** Achieve a lift-to-drag ratio of at least 20 for a generated design.
*   **Computational Cost:** Reduce the training time to under 24 hours on a single GPU.
*   **Design Diversity:** Generate a set of at least 100 unique and viable aircraft designs.

## 4. Suggestions for Further Research

*   **Structural Analysis:** Integrate a structural analysis solver (e.g., finite element analysis) into the generative design framework to evaluate the structural integrity of the generated designs.
*   **Manufacturability:** Incorporate manufacturability constraints into the generative process to ensure that the generated designs can be easily fabricated.
*   **Multi-Disciplinary Optimization:** Extend the framework to perform multi-disciplinary optimization, considering other aspects of aircraft design, such as acoustics and thermodynamics.
*   **Advanced CFD Solvers:** Integrate a more advanced CFD solver into the framework, such as a Reynolds-Averaged Navier-Stokes (RANS) solver or a Large Eddy Simulation (LES) solver.
