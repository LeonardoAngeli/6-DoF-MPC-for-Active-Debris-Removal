<a id="readme-top"></a>

<div align="center">

  <h1>MPC for Active Debris Removal</h1>
  <h3>Autonomous rendezvous with an uncooperative SL-16 upper stage</h3>

  <p>
    Linear MPC · Kalman filtering · Nonlinear J2 plant · 1000-run Monte Carlo
  </p>

  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy">
  <img src="https://img.shields.io/badge/CVXPY-Clarabel-4B8BBE?style=for-the-badge" alt="CVXPY">
  <img src="https://img.shields.io/badge/status-Phase%202%20of%205-orange?style=for-the-badge" alt="Status">

  <br /><br />
  <img src="figures/mc_trajectories.png" alt="Monte Carlo trajectories" width="600">

</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About the project</a></li>
    <li><a href="#scenario">Scenario</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#mpc-formulation">MPC formulation</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#analyses-and-lessons-learned">Analyses and lessons learned</a></li>
    <li><a href="#assumptions-and-limitations">Assumptions and limitations</a></li>
    <li><a href="#getting-started">Getting started</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#note-on-ai-assistance">Note on AI assistance</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


## About the project

<!-- 3–5 sentences: the ADR problem, why an SL-16, what this repository does
     (linear MPC + KF on a nonlinear J2 plant), and where it is in the 5-phase plan. -->

**Highlights**

* 1000-run Monte Carlo: **0 failures**, keep-out never violated, all runs settled
* Multi-rate architecture: MPC at 30 s, plant and filter at 1 s (30× smaller QP, same Δv)
* Measured validity envelope of the linear model (thrust authority and linearization bias)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Scenario

| | |
|---|---|
| Target | SL-16 upper stage, LEO, 843.5 km, i ≈ 71° |
| Chaser | 500 kg, 1 N-class monopropellant RCS → u<sub>max</sub> = 2·10⁻³ m/s² |
| Initial condition | co-elliptic drift, 100–300 m below and 800–1200 m behind the target |
| Final condition | hold point on V-bar, 30 m behind the target |
| Out of scope | phasing (electric propulsion), attitude control, final capture |

<!-- 1–2 sentences on why a co-elliptic approach from behind. -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Architecture