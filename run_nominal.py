# %%
# %load_ext autoreload
# %autoreload 2

import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy

from dynamics.ss_equations import SchweighartSedwickDynamics
from dynamics.nonlinear_plant import NonLinearPlant
from sim.simulation import run_simulation_NL
from sim.model_noise import ModelNoise
from sim.sensor_noise import SensorNoise
from estimation.kalman_filter import KalmanFilter
from mpc.mpc_3dof import LinearMpc


# %%

"Nominal scenario: approach from below and behind the target"

# LVLH frame:
#  - x: radial direction, from the Earth to the target
#  - y: along-track direction
#  - z: direction normal to the orbital plane

# Target orbit (SL-16 upper stage)
mu = 3.986004418e14         # m^3/s^2, Earth gravitational parameter
Re = 6378137.0              # m, Earth radius
J2 = 1.08263e-3             # Earth oblateness coefficient
i = 70.9997                 # deg, inclination
e = 0.0003119               # eccentricity
omega = 54.2008             # deg, argument of perigee
RAAN = 138.4406             # deg, right ascension of the ascending node
theta = 305.91              # deg, true anomaly
a = Re + 843.5e3            # m, semi-major axis
n = np.sqrt(mu / a**3)      # rad/s, mean motion

# Time steps:
# - Plant, sensor and KF at dt; 
# - MPC at dt_mpc = mult_rate * dt
dt = 1.0
mult_rate = 30
dt_mpc = dt * mult_rate
N = 6000                    # N. simulation steps

seed = 0
rng = np.random.default_rng(seed)

# Initial true state: co-elliptic drift, delta_h = 200 m below the target
# For a circular orbit lower by delta_h: x = -delta_h, y_dot = 1.5 * n * delta_h
delta_h = 200.0
x0 = np.array([-delta_h, -1000.0, 0.0, 0.0, 1.5 * n * delta_h, 0.0])
u_prev = np.zeros(3)


# %%

"Linear models (Schweighart-Sedwick)"

# KF, process noise and residual analysis use dt; MPC uses dt_mpc
dyn = SchweighartSedwickDynamics(n, dt, J2, Re, a, np.radians(i))
dyn_mpc = SchweighartSedwickDynamics(n, dt_mpc, J2, Re, a, np.radians(i))
Ad, Bd = dyn.get_discrete_matrices()
Ad_mpc, Bd_mpc = dyn_mpc.get_discrete_matrices()


# %%

"Process noise, LIDAR sensor and Kalman filter"

# Process disturbance on the true plant
sigma_a = 1e-5              # m/s^2
noise = ModelNoise(Bd, sigma_a, rng)

# Initial estimation error
P0 = np.diag([50.0**2, 100.0**2, 50.0**2] + [0.05**2] * 3)
x0_kf = rng.multivariate_normal(x0, P0)

# LIDAR: relative position only, range-dependent noise
C = np.hstack([np.eye(3), np.zeros((3, 3))])
sensor = SensorNoise(C, rng)

# KF process noise inflated (x9) to cover unmodelled dynamics (linearization error, see README)
Q = 3**2 * noise.Q
kf = KalmanFilter(Ad, Bd, C, Q, sensor, x0_kf, P0)


# %%

"Linear MPC"

# Target point 30 m behind the target
x_ref = np.array([0.0, -30.0, 0.0, 0.0, 0.0, 0.0])

# Weights (Bryson's rule)
x_tol = np.array([3.0, 3.0, 10.0])         # position tolerance (design choice)
v_tol = np.array([0.05, 0.05, 0.05])       # velocity tolerance (design choice)
u_max = 2e-3                               # m/s^2, maximum thrust acceleration

Q_mpc = np.diag(np.concatenate([1 / x_tol**2, 1 / v_tol**2]))
R_mpc = np.eye(3) / u_max**2

# Terminal cost
P_mpc = scipy.linalg.solve_discrete_are(Ad_mpc, Bd_mpc, Q_mpc, R_mpc)
P_mpc = 0.5 * (P_mpc + P_mpc.T)

# Constraints
D_u_max = mult_rate * u_max / 25               # max thrust change 
R_ko = 20.0                                    # m, keep-out half-space distance
n_ko = x_ref[:3] / np.linalg.norm(x_ref[:3])   # keep-out normal, same as in LinearMpc

# Horizon: Hp * dt_mpc seconds
Hp = 20

# Soft-constraint penalties
rho = 1e3                                  # keep-out
rho_v = 2e5                                # speed limit

mpc = LinearMpc(Ad_mpc, Bd_mpc, x_ref, Q_mpc, R_mpc, P_mpc, u_max, D_u_max, R_ko, Hp, rho, rho_v, dt_mpc)


# %%

"Simulation"

plantNL = NonLinearPlant(a, dt, J2, Re, i, e, omega, RAAN, theta, mu, x0)
trajectory_KF, trajectory_true, measures, P_v, u_total_v, mpc_log = run_simulation_NL(
    plantNL, noise, sensor, kf, mpc, x0, N, u_prev, mult_rate)


# %%

"Metrics"

X = trajectory_true
U = u_total_v[:-1]                         
t = np.arange(N + 1) * dt

# Fuel
dv = np.sum(np.linalg.norm(U, axis=1)) * dt

# Safety: distance to target and keep-out half-space margin 
dist = np.linalg.norm(X[:, :3], axis=1)
wall_margin = X[:, :3] @ n_ko - R_ko       # >= 0 means the half-space is respected
speed = np.linalg.norm(X[:, 3:], axis=1)

# Last time outside |pos - pos_ref| < 1 m or |vel| < 0.01 m/s
pos_err = np.linalg.norm(X[:, :3] - x_ref[:3], axis=1)
vel_err = np.linalg.norm(X[:, 3:] - x_ref[3:], axis=1)
outside = np.where((pos_err > 1.0) | (vel_err > 0.01))[0]
settled = outside.size == 0 or outside[-1] < N
t_settle = (outside[-1] + 1) * dt if outside.size else 0.0

# KF consistency: fraction of samples inside the 3-sigma bound
error = X - trajectory_KF
sigma = np.sqrt(np.diagonal(P_v, axis1=1, axis2=2))
frac = (np.abs(error) < 3 * sigma).mean(axis=0)


print("---- Fuel ----")
print(f"Total delta-v [m/s]:            {dv:.3f}")
print("---- Safety ----")
print(f"Min distance to target [m]:     {dist.min():.2f}")
print(f"Min keep-out margin [m]:        {wall_margin.min():.2f}")
print(f"Max speed [m/s]:                {speed.max():.3f}   (v_max = {mpc.v_max:.3f})")
print(f"Max keep-out slack eps [m]:     {mpc_log['eps'].max():.2e}")
print(f"Max speed slack eps_v [m/s]:    {mpc_log['eps_v'].max():.2e}")
print("---- Performance ----")
print(f"Settling time [s]:              {t_settle:.0f}" + ("" if settled else "   (NOT settled)"))
print(f"Final position error [m]:       {pos_err[-1]:.3f}")
print(f"Final velocity error [m/s]:     {vel_err[-1]:.4f}")
print("---- Solver ----")
print(f"MPC solves:                     {len(mpc_log['status'])}")
print(f"Iterations (max):               {mpc_log['iters'].max()}")
print(f"Solve time max / p99 [ms]:      {1e3 * mpc_log['t_solve'].max():.1f} / {1e3 * np.percentile(mpc_log['t_solve'], 99):.1f}")
print("---- Navigation ----")
print("Fraction inside 3-sigma:       ", np.round(frac, 4))


# %%

"Plots"

os.makedirs("figures", exist_ok=True)

# Trajectory in the orbital plane
plt.plot(trajectory_true[:, 0], trajectory_true[:, 1], label="true")
plt.plot(trajectory_KF[:, 0], trajectory_KF[:, 1], "--", label="KF estimate")
plt.gca().add_patch(plt.Circle((0, 0), R_ko, color="r", alpha=0.3, label="keep-out"))
plt.axhline(-R_ko, color="r", linestyle=":", label="keep-out half-space")
plt.plot(x_ref[0], x_ref[1], "kx", label="reference")
plt.xlabel("x (radial) [m]")
plt.ylabel("y (along-track) [m]")
plt.axis("equal")
plt.legend()
plt.grid()
plt.savefig("figures/trajectory_xy.png")
plt.show()

# Thrust command
plt.plot(t, u_total_v[:, 0], label="u_x")
plt.plot(t, u_total_v[:, 1], label="u_y")
plt.plot(t, u_total_v[:, 2], label="u_z")
plt.hlines([u_max, -u_max], 0, N * dt, "r", "--", label="u_max")
plt.xlabel("t [s]")
plt.ylabel("u [m/s^2]")
plt.legend()
plt.grid()
plt.savefig("figures/thrust.png")
plt.show()

# Distance to target
plt.plot(t, dist)
plt.hlines(R_ko, 0, N * dt, "r", "--", label="R_ko")
plt.xlabel("t [s]")
plt.ylabel("distance [m]")
plt.legend()
plt.grid()
plt.savefig("figures/distance.png")
plt.show()

# Speed vs speed limit
plt.plot(t, speed)
plt.hlines(mpc.v_max, 0, N * dt, "r", "--", label="v_max")
plt.xlabel("t [s]")
plt.ylabel("|v| [m/s]")
plt.legend()
plt.grid()
plt.savefig("figures/speed.png")
plt.show()

# KF estimation error with 3-sigma bound (for x)
plt.plot(t, error[:, 0], label="error x")
plt.plot(t, 3 * sigma[:, 0], "r--", label="±3σ")
plt.plot(t, -3 * sigma[:, 0], "r--")
plt.xlabel("t [s]")
plt.ylabel("error x [m]")
plt.legend()
plt.grid()
plt.savefig("figures/kf_error_x.png")
plt.show()

# LIDAR noise standard deviation vs time
plt.plot(np.array(sensor.sigma_p_v))
plt.hlines(sensor.sigma_p0, 0, N, "r", "--", label="sigma_0")
plt.xlabel("t [s]")
plt.ylabel("sigma_p [m]")
plt.legend()
plt.grid()
plt.savefig("figures/lidar_sigma.png")
plt.show()