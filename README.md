# MPC for Autonomous Rendezvous with an Uncooperative SL-16 Upper Stage

One-paragraph summary: what the project is (ADR rendezvous, linear MPC + KF,
nonlinear J2 plant), current status (Phase 2 of 5: translational control),
headline result (1000-run Monte Carlo, 0 failures).

## Scenario
- Target: SL-16 upper stage, LEO (843.5 km, i ≈ 71°)
- Chaser: 500 kg, 1 N-class monopropellant RCS → u_max = 2e-3 m/s²
- Approach: co-elliptic drift from 100–300 m below and ~1 km behind,
  to a hold point on V-bar 30 m behind the target
- Out of scope: phasing (electric propulsion), attitude, final capture

## Architecture
(Diagram or list) Nonlinear plant (two-body + J2) → LIDAR sensor → Kalman filter
→ MPC (every 30 s) → thrust (ZOH)

## Models
- Plant: nonlinear two-body + J2, both bodies propagated in ECI (DOP853)
- Controller/filter model: Schweighart-Sedwick (linearized, mean J2), ZOH
- Sensor: LIDAR relative position, σ(r) = 0.1 m + 0.1 % · r / 3
  (Jena-Optronik RVS3000-3D datasheet, floor is an assumption)

## MPC formulation
- Cost, constraints (copy the formulation block from mpc_3dof.py)
- Design choices and why:
  - Bryson's rule for Q, R; DARE terminal cost
  - soft constraints with exact L1 penalty (feasibility under noise)
  - fixed keep-out half-space normal (deadlock with a moving normal)
  - speed limit tied to the horizon: v_max = 0.8 u_max Hp dt
  - multi-rate: MPC at 30 s, plant/KF at 1 s
  - constraint tightening: +1 m on the keep-out (estimation error)

## Results
### Nominal run
(figures: trajectory_xy.png, thrust.png, speed.png + key numbers)
### Monte Carlo (1000 runs)
(table: Δv, settling time, max speed — mean / p95 / max)
- worst keep-out margin +0.83 m, 1000/1000 settled
- 0 solver failures → failure probability < 0.3 % (95 % confidence)
- KF consistency: mean 0.9975 (ideal 0.9973), worst 0.979
(figures: mc_trajectories.png, mc_dv_hist.png)

## Analyses and lessons learned
- Multi-rate: same Δv as 1 s MPC with a 30× smaller problem
- Validity envelope: thrust authority u_max/(3n²); linearization bias
  beyond ~1 km breaks KF consistency (needs ~12× process noise)
- KF tuning: thrust-dependent unmodelled dynamics during manoeuvres
- Speed limit: below the exact-penalty threshold the soft constraint is ignored

## Assumptions and limitations
- KF Q, R and C (relative position only) are declared assumptions
- Perfect attitude, omnidirectional thrust, non-tumbling target
- No minimum impulse bit / thrust modulation
- Process noise inflated ×9 to cover unmodelled dynamics

## How to run
pip install -r requirements.txt
python run_nominal.py
python run_monte_carlo.py   (~6 h for 1000 runs)

## Roadmap
Phase 3–5: attitude dynamics, attitude MPC, coupled nonlinear 6-DoF MPC.
Next on Phase 2: unit tests, sensitivity study, design speed cap.

## Note on AI assistance
(Your own words: what you did, what you used Claude for.)