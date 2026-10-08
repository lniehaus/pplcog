# Parameters of the closed-loop simulation

Parameters of the closed-loop pitch-plane simulation (Section 2.5 and Figures 5 and 6 of the
manuscript *Adaptive Center of Gravity Control for Push-Pull Locomotion Rovers on Slopes*),
in addition to those of the model evaluation in `parameters_model_evaluation.md`. This table
was Table 2 of the first submission; the revised manuscript refers to this file from its
Data Availability Statement. It is written by `pplcog/sim/run_sim.py` (full run only) from the defaults in
`pplcog/sim/params_sim.py`; edit the defaults, not this file.

Sources as in `parameters_model_evaluation.md`. The Wong-Reece coefficients follow
Wong and Reece (1967) and Ishigami et al. (2007). Soil parameters are randomised per soil
sample; a pair of values denotes the sampled range.

| Parameter | Value | Unit | Source | Note |
|---|---|---|---|---|
| a_0 | 0.4 | - | literature | Wong and Reece 1967: 0.18 (loose) to 0.43 (compact) sand; sampled 0.2 to 0.45, assumed |
| a_1 | 0.15 | - | literature | Ishigami et al. 2007: 0 to 0.3; Wong and Reece 1967 report 0.32; sampled 0.1 to 0.35, assumed |
| \lambda | 0.1 | - | assumed | rear exit sinkage ratio |
| \gamma_s | 1.28e+04 | N/m^3 | literature | converted from 1.29 to 1.31 g/cm3, silica sand No. 5, SFI25 Table 4 |
| K | (0.01, 0.025) | m | literature | shear deformation modulus of sand, 10 to 25 mm (Wong 2008); clay about 6 mm |
| \Delta t | 0.001 | s | study | integration step |
| cycles | 10 | - | manuscript | ten cycles per run, validation protocol |
| s_{ref} | 0.15 | - | assumed | commanded slip of the driven pair |
| K_{p,L} | 4000 | N/m | assumed | wheelbase actuator position loop |
| K_{d,L} | 120 | N s/m | assumed |  |
| F_{act,max} | 120 | N | assumed | wheelbase actuator force limit |
| \dot L_{max} | 0.15 | m/s | assumed | wheelbase rate limit |
| T_{max} | 3 | N m | assumed | motor torque limit per wheel |
| \zeta_a | 0.7 | - | assumed | anchor damping ratio |
| \tau_a | 0.05 | s | assumed | load-transfer acceleration filter |
| v_{slide} | 0.05 | m/s | study | backward sliding criterion |
| \eta_w | 0.7 | - | assumed | drive efficiency |
| \eta_a | 0.7 | - | assumed | wheelbase actuator efficiency |
| \eta_r | 0.7 | - | assumed | rail actuator efficiency |
| s_{stall} | 0.95 | - | study | stall criterion |
| \psi_{max} | 0.2618 | rad | study | pitch failure criterion |
| \Delta_{max}/R\ \mathrm{cap} | 0.5 | - | assumed | design-time cap on the envelope |
| seed | 20261002 | - | study |  |
| n_{soil} | 60 | - | study | soil population |
| n_{train} | 40 | - | study | training soils |
| n_{rep} | 5 | - | manuscript | repetitions per condition, validation protocol |
| classes | 3 | - | manuscript | terrain classes for Stage A |
| \ell_{min} | 0.5 | - | study | lower bound of the Stage C Gaussian-process length scales, in standard deviations of each input |
