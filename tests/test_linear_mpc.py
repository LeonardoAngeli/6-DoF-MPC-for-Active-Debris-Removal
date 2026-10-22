# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy

from dynamics.ss_equations import SchweighartSedwickDynamics
from dynamics.orbit_propagator import OrbitPropagator
from dynamics.nonlinear_plant import NonLinearPlant
from sim.simulation import run_simulation_NL
from sim.model_noise import ModelNoise
from sim.sensor_noise import SensorNoise
from estimation.kalman_filter import KalmanFilter
from mpc.mpc_3dof import LinearMpc



# %%

"Test the linear MPC"

# Initial positions and speed of the Chaser
x0 = np.array([
    500,
    100.0,           
    0.0,            
    0.3,
    1.75,
    0.1
])

# Note:
#  - x: radial directon b/w the earth and the target satellite
#  - y: along-track direction, perpendicular to the radial direction
#  - z: cross-track direction, perpendicular to the orbital plane

u_prev = np.zeros(3)

# Define the motion and time step
mu = 3.986004418e14    # m^3/s^2, Gravitational parameter Earth
Re = 6378137.0          # m, Earth Radius
J2 = 1.08263e-3         # adimensional
i = 70.9997                  # inclination
e = 0.0003119             # eccentricity
omega = 54.2008             # argument of the perigee
RAAN = 138.4406               # RAAN
theta = 305.91               # anomaly
a = Re + 843.5e3               # semi-major axis of the orbit

n = np.sqrt(mu / a**3)
dt = 1.0


## Setting up the structure for simulation ##
seed = 3
rng = np.random.default_rng(seed)

# Plant (SS for the KF)
dynamics = SchweighartSedwickDynamics(n, dt, J2, Re, a, np.radians(71))
Ad, Bd = dynamics.get_discrete_matrices()
plant = OrbitPropagator(dynamics)


# Sensors and Noise
sigma_a = 1e-5
sigma_p = 10

# Assuming our sensor gives us the relative position with the target
C = np.hstack([np.eye(3), np.zeros((3,3))])
noise = ModelNoise(Bd,sigma_a,rng)
sensor = SensorNoise(sigma_p,C,rng)

# Kalman Filter
x0_kf = x0.copy()


# Q_mis from mismatch in the model
Q_mis = np.array([
    [ 1.74346529e-10, -3.41283948e-12, -2.70186973e-13,  3.48694064e-10, -6.88517274e-12, -5.40307120e-13],
    [-3.41283948e-12,  7.29593737e-11, -2.53541307e-13, -6.84302738e-12,  1.45925006e-10, -5.07124040e-13],
    [-2.70186973e-13, -2.53541307e-13,  5.62040369e-14, -5.40388497e-13, -5.06915427e-13,  1.12410690e-13],
    [ 3.48694064e-10, -6.84302738e-12, -5.40388497e-13,  6.97390149e-10, -1.38050431e-11, -1.08064333e-12],
    [-6.88517274e-12,  1.45925006e-10, -5.06915427e-13, -1.38050431e-11,  2.91862563e-10, -1.01391374e-12],
    [-5.40307120e-13, -5.07124040e-13,  1.12410690e-13, -1.08064333e-12, -1.01391374e-12,  2.24827522e-13],
])

# Total Q for KF
Q = noise.Q + Q_mis

P0 = np.diag([10**2]*3 + [0.1**2]*3)   # Test
 # P0 = np.diag([1000**2]*3 + [30**2]*3)   # σ_pos0 = 5 m, σ_vel0 = 0.5 m/s
kf = KalmanFilter(Ad, Bd, C, Q, sensor.R, x0_kf, P0)

# MPC

# Input Mpc
x_ref = np.array([0, 30, 0, 0, 0, 0])

# Weights
B = 1.0
D = 25 * B
A = 1.3287 * B
Q_mpc = np.diag([B, B/4, B, D, D/4, D])
R_mpc = A * np.eye(3)
P_mpc = scipy.linalg.solve_discrete_are(Ad, Bd, Q_mpc, R_mpc)   # Ad, Bd 
P_mpc = 0.5 * (P_mpc + P_mpc.T)

# Constraints
u_max = 0.00425
D_u_max = u_max / 10
R_ko = 20.0

# Prediction Horizon
Hp = 200

# Slack variable
epsilon = 1
rho = 1e3
rho_v = 500

# mpc
mpc = LinearMpc(Ad, Bd, x_ref, Q_mpc, R_mpc, P_mpc, u_max, D_u_max, R_ko, Hp,rho,rho_v,dt)


# plant NL 
plantNL = NonLinearPlant(a,dt,J2,Re,i,e,omega,RAAN,theta,mu,x0)

u0 = mpc.compute_control(x0, np.zeros(3))
print(type(u0), mpc.prob.status, getattr(mpc, "u_opt", None))

# Simulation
N = 6000  # Number of time steps
trajectory_KF, trajectory_true, measures, P_v = run_simulation_NL(plantNL, noise, sensor, kf, mpc, x0, N, u_prev)

# %%

"Compare the results"

error = trajectory_true - trajectory_KF
sigma = np.sqrt(np.diagonal(P_v, axis1=1, axis2=2))     # Variance from matrix pile to matrix
frac = (np.abs(error) < 3*sigma).mean(axis=0)           # Under 3*sigma

print("Fraction inside 3_sigma:", frac)
# assert np.all(frac > 0.99)                              # Fail-safe


"Plot the results for visualization"
t = np.arange(N + 1) * dt

plt.plot(trajectory_KF[:, 0], trajectory_KF[:, 1])
plt.plot(trajectory_true[:, 0], trajectory_true[:, 1])
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.grid()
plt.show()


plt.plot(t, trajectory_true[:, 0])
plt.ylim(np.min(trajectory_true[:, 0]), np.max(trajectory_true[:, 0]))
plt.hlines(x_ref[0],0,N,"r",'--')
plt.xlabel("t [s]")
plt.ylabel("x [m]")
plt.grid()
plt.show()

plt.plot(t, trajectory_true[:, 1])
plt.ylim(np.min(trajectory_true[:, 1]), np.max(trajectory_true[:, 1]))
plt.hlines(x_ref[1],0,N,"r",'--')
plt.xlabel("t [s]")
plt.ylabel("y [m]")
plt.grid()
plt.show()


plt.plot(t, error[:, 0], label="error x")
plt.plot(t, 3*sigma[:, 0], "r--", label="+3σ")
plt.plot(t, -3*sigma[:, 0], "r--")
plt.xlabel("t [s]")
plt.ylabel("error x [m]")
plt.legend()
plt.grid()
plt.show()



# %%

