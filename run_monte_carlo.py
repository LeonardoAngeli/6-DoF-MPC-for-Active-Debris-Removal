# %%
import numpy as np
import scipy as scipy

from dynamics.ss_equations import SchweighartSedwickDynamics
from dynamics.nonlinear_plant import NonLinearPlant
from sim.simulation import run_simulation_NL
from sim.model_noise import ModelNoise
from sim.sensor_noise import SensorNoise
from estimation.kalman_filter import KalmanFilter
from mpc.mpc_3dof import LinearMpc


# %%

"Monte Carlo Sims"

# Target orbit (SL-16 upper stage)
mu = 3.986004418e14
Re = 6378137.0
J2 = 1.08263e-3
i = 70.9997
e = 0.0003119
omega = 54.2008
RAAN = 138.4406
theta = 305.91
a = Re + 843.5e3
n = np.sqrt(mu / a**3)

# Time steps
dt = 1.0
mult_rate = 30
dt_mpc = dt * mult_rate
N = 6000

# Linear models
dyn = SchweighartSedwickDynamics(n, dt, J2, Re, a, np.radians(i))
dyn_mpc = SchweighartSedwickDynamics(n, dt_mpc, J2, Re, a, np.radians(i))
Ad, Bd = dyn.get_discrete_matrices()
Ad_mpc, Bd_mpc = dyn_mpc.get_discrete_matrices()

# Navigation
sigma_a = 1e-5
P0 = np.diag([50.0**2, 100.0**2, 50.0**2] + [0.05**2] * 3)
C = np.hstack([np.eye(3), np.zeros((3, 3))])

# MPC (same as run_nominal.py)
x_ref = np.array([0.0, -30.0, 0.0, 0.0, 0.0, 0.0])
x_tol = np.array([3.0, 3.0, 10.0])
v_tol = np.array([0.05, 0.05, 0.05])
u_max = 2e-3
Q_mpc = np.diag(np.concatenate([1 / x_tol**2, 1 / v_tol**2]))
R_mpc = np.eye(3) / u_max**2
P_mpc = scipy.linalg.solve_discrete_are(Ad_mpc, Bd_mpc, Q_mpc, R_mpc)
P_mpc = 0.5 * (P_mpc + P_mpc.T)
D_u_max = mult_rate * u_max / 25
R_ko = 20.0
ko_margin = 1.0
n_ko = x_ref[:3] / np.linalg.norm(x_ref[:3])
Hp = 20
rho = 1e3
rho_v = 2e5


# %%

"Runs"

n_runs = 50

dv_v = []
t_settle_v = []
margin_v = []
speed_v = []
frac_v = []
failures = []
traj_v = []

for seed in range(n_runs):
    rng = np.random.default_rng(seed)

    # Dispersed initial state:
    # - depth dh in [100, 300] m
    # - along-track distance in [-1200, -800] m
    # - out-of-plane offset, 30 m 
    # - arrival velocity error, 0.02 m/s per axis

    dh = rng.uniform(100.0, 300.0)
    x0 = np.array([-dh, rng.uniform(-1200.0, -800.0), rng.normal(0.0, 30.0),
                   0.0, 1.5 * n * dh, 0.0])
    x0[3:] += rng.normal(0.0, 0.02, 3)


    noise = ModelNoise(Bd, sigma_a, rng)
    x0_kf = rng.multivariate_normal(x0, P0)
    sensor = SensorNoise(C, rng)
    kf = KalmanFilter(Ad, Bd, C, 3**2 * noise.Q, sensor, x0_kf, P0)
    mpc = LinearMpc(Ad_mpc, Bd_mpc, x_ref, Q_mpc, R_mpc, P_mpc, u_max, D_u_max,
                    R_ko + ko_margin, Hp, rho, rho_v, dt_mpc)
    plantNL = NonLinearPlant(a, dt, J2, Re, i, e, omega, RAAN, theta, mu, x0)

    try:
        trajectory_KF, trajectory_true, measures, P_v, u_total_v, mpc_log = run_simulation_NL(
            plantNL, noise, sensor, kf, mpc, x0, N, np.zeros(3), mult_rate)
    except RuntimeError:
        print(f"run {seed}: SOLVER FAILURE")
        failures.append(seed)
        continue

    X = trajectory_true
    U = u_total_v[:-1]

    # Fuel
    dv_v.append(np.sum(np.linalg.norm(U, axis=1)) * dt)

    # Last time outside 1 m / 0.01 m/s from the reference
    pos_err = np.linalg.norm(X[:, :3] - x_ref[:3], axis=1)
    vel_err = np.linalg.norm(X[:, 3:] - x_ref[3:], axis=1)
    outside = np.where((pos_err > 1.0) | (vel_err > 0.01))[0]
    t_settle_v.append((outside[-1] + 1) * dt)

    # Safety
    margin_v.append(np.min(X[:, :3] @ n_ko - R_ko))
    speed_v.append(np.max(np.linalg.norm(X[:, 3:], axis=1)))

    # KF consistency (worst state)
    error = X - trajectory_KF
    sigma = np.sqrt(np.diagonal(P_v, axis1=1, axis2=2))
    frac_v.append(np.min((np.abs(error) < 3 * sigma).mean(axis=0)))

    # Trajectory
    traj_v.append(X[::10, :3])

    print(f"run {seed}: dv = {dv_v[-1]:.2f} m/s, settle = {t_settle_v[-1]:.0f} s, margin = {margin_v[-1]:.2f} m")

    # Save partial results after every run
    np.savez("monte_carlo_results_traj.npz", dv=dv_v, t_settle=t_settle_v,
    margin=margin_v, speed=speed_v, frac=frac_v, traj=np.array(traj_v))

# %%

"Results"

dv_v = np.array(dv_v)
t_settle_v = np.array(t_settle_v)
margin_v = np.array(margin_v)
speed_v = np.array(speed_v)
frac_v = np.array(frac_v)

print("Delta-v [m/s]:        mean", np.round(dv_v.mean(), 2), "  p95", np.round(np.percentile(dv_v, 95), 2), "  max", np.round(dv_v.max(), 2))
print("Settling time [s]:    mean", np.round(t_settle_v.mean()), "  p95", np.round(np.percentile(t_settle_v, 95)), "  max", np.round(t_settle_v.max()))
print("Max speed [m/s]:      mean", np.round(speed_v.mean(), 2), "  max", np.round(speed_v.max(), 2), "  theorical", np.round(mpc.v_max, 2))
print("Worst keep-out margin [m]:", np.round(margin_v.min(), 2))
print("Mean KF consistency:      ", np.round(frac_v.mean(), 4))
print("Worst KF consistency:     ", np.round(frac_v.min(), 4))
print("Runs settled within N:    ", np.sum(t_settle_v < N * dt), "/", n_runs)
print("Runs with margin >= 0:    ", np.sum(margin_v >= 0), "/", n_runs)
print("Solver failures:          ", len(failures), "/", n_runs, failures)

# %% 

import matplotlib.pyplot as plt

"Plots"

# Delta-v distribution (1000 runs)
data = np.load("monte_carlo_results.npz")
plt.hist(data["dv"], bins = 70)
plt.xlabel("Δv [m/s]")
plt.ylabel("number of runs")
plt.grid()
plt.savefig("figures/mc_dv_hist.png")
plt.show()

traj = np.load("monte_carlo_results_traj.npz")["traj"]
for T in traj:
    plt.plot(T[:, 0], T[:, 1], color="C0", linewidth=0.5, alpha=0.5)
plt.gca().add_patch(plt.Circle((0, 0), R_ko, color="r", alpha=0.3, label="keep-out"))
plt.axhline(-R_ko, color="r", linestyle=":", label="keep-out half-space")
plt.plot(x_ref[0], x_ref[1], "kx", label="reference")
plt.xlabel("x (radial) [m]")
plt.ylabel("y (along-track) [m]")
plt.axis("equal")
plt.legend()
plt.grid()
plt.savefig("figures/mc_trajectories.png")
plt.show()



# %%
