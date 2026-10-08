# Parameters of the model evaluation

Every parameter of the quasi-static model evaluation (Section 3.1.2 and Figure 4 of the
manuscript *Adaptive Center of Gravity Control for Push-Pull Locomotion Rovers on Slopes*)
with its value, unit, source, and note. This table was Table 3 of the first submission; the revised manuscript
refers to this file from its Data Availability Statement. It is written by `pplcog/run_all.py` from
the defaults in `pplcog/params.py`; edit the defaults, not this file.

Sources:

- *photo* = read from the labelled scale of Figure 1(a) of the manuscript;
- *sibling* = published rovers of the same laboratory (Higa, Fujiwara and Iizuka 2024, HFI24;
  Sugimoto, Fujiwara and Iizuka 2025, SFI25);
- *literature* = silica sand No. 5 properties (SFI25), Bekker constants for dry sand
  (Wong 2008, *Theory of Ground Vehicles*, Wo08), and the rigid-wheel model of
  Wong and Reece (1967, WR67);
- *assumed* = not available for the present build;
- *manuscript* = design choice stated in the text;
- *study* = design of this evaluation.

All photo and assumed values are estimates with an uncertainty of about +-20 %.

| Parameter | Value | Unit | Source | Note |
|---|---|---|---|---|
| M | 12 | kg | sibling | SFI25 rover 120.5 N (12.3 kg); illustrative only |
| f = m_m/M | 0.3 | - | manuscript | 'substantial fraction'; swept 0.2 to 0.5 |
| m_{pair}/M | 0.12 | - | assumed | each wheel pair with motors |
| L_{max} | 0.52 | m | photo | wheel-centre spacing from the Fig. 1a scale, +-20 % |
| L_{max}-L_{min} | 0.12 | m | sibling | HFI24 120 mm; SFI25 80 mm |
| r | 0.135 | m | photo | wheel diameter 250 to 300 mm from the Fig. 1a scale |
| b | 0.1 | m | assumed | wheel width, not visible in the side view |
| rail frame | rear | - | assumed | chassis and rail fixed to this pair; both evaluated |
| \xi_{rail,0} | 0.116 | m | assumed | rail centred on the chassis at L_max; photo shows the rail zero near one axle, swept 0.05 to 0.182 |
| R | 0.288 | m | photo | labelled 0, 96, 192, 288 mm |
| \xi_b | 0.26 | m | assumed | body CoG at mid-chassis at L_max |
| h_b | 0.26 | m | assumed | about wheel top height |
| h_{rail} | 0.38 | m | photo | +-20 %; sensitivity sweep 0.5 to 0.9 L |
| N_{min}/(Mg) | 0.1 | - | manuscript | design choice for the stability margin |
| g | 9.81 | m/s^2 | literature |  |
| k_c | 990 | N/m^(n+1) | literature | Wong 2008 dry sand; fallback, not measured for silica sand No. 5 |
| k_\phi | 1.53e+06 | N/m^(n+2) | literature | Wong 2008 dry sand; fallback |
| n | 1.1 | - | literature | Wong 2008 dry sand; fallback |
| c | 761.8 | Pa | literature | silica sand No. 5, SFI25 Table 4 (unit printed as N/m3, read as Pa) |
| \phi_s | 0.3892 | rad | literature | silica sand No. 5, 22.3 deg, SFI25 Table 4 after Matsumoto 2013 |
| K | 0.01 | m | literature | Wong 2008 range 0.001 to 0.025 m as quoted in SFI25 Table 4, log-midpoint |
| s_{ref} | 0.5 | - | assumed | slip at which H_max is evaluated |
| \tau | 0.3 | s | assumed | stepper position loop |
| v_{max} | 0.3 | m/s | assumed | mass crosses the rail within one phase |
| a_{max} | 2 | m/s^2 | assumed |  |
| T_{phase} | 2 | s | assumed |  |
| \Delta t | 0.002 | s | study | integration step |
| K_p | 0.1 | m | assumed | PI on the load fraction |
| K_i | 0.2 | m/s | assumed | PI on the load fraction |
| \delta x_b/L | 0.03 | - | study | plant mismatch that gives the PI a task |
| \theta_{ref} | 20 | deg | study |  |
| \rho^\ast | 1 | - | manuscript | target load ratio |
| seed | 20261001 | - | study | Monte Carlo seed |
| n_{MC} | 200 | - | study | soil and geometry samples |
| cycles | 3 | - | study | simulated PPL cycles |
