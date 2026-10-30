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

iterations = 1
frac_v = []
for seed in range(iterations):
    print(seed)


    "Test the linear MPC"

    # Initial positions and speed of the Chaser
    x0 = np.array([
        -200,
        -1000,           
        0,            
        0.0,
        0.31,
        0
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
    mult_rate = 60
    dt_mpc = dt * mult_rate

    ## Setting up the structure for simulation ##
    rng = np.random.default_rng(seed)


    # Plant (SS)
    dyn    = SchweighartSedwickDynamics(n, dt,     J2, Re, a, np.radians(71))
    dyn_mpc = SchweighartSedwickDynamics(n, dt_mpc, J2, Re, a, np.radians(71))
    Ad, Bd = dyn.get_discrete_matrices()               # KF, noise, residual W
    Ad_mpc, Bd_mpc = dyn_mpc.get_discrete_matrices()   # MPC and DARE only
    plant = OrbitPropagator(dyn)


    # Sensors and Noise
    sigma_a = 1e-5

    # Kalman Filter
    x0_kf = np.array([
        -200,
        -1000,           
        0,            
        0,
        0.31,
        0
    ])


    # Assuming our sensor gives us the relative position with the target
    C = np.hstack([np.eye(3), np.zeros((3,3))])
    noise = ModelNoise(Bd,sigma_a,rng)
    sensor = SensorNoise(C,rng,x0_kf)


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
    Q = 3**2 * noise.Q 

    # P0 = np.diag([10**2]*3 + [0.1**2]*3)   # Test
    P0 = np.diag([500**2]*3 + [0.5**2]*3)     # σ_pos0 = 5 m, σ_vel0 = 0.5 m/s
    kf = KalmanFilter(Ad, Bd, C, Q, sensor, x0_kf, P0)

    # MPC

    # Input Mpc
    x_ref = np.array([0, 30, 0, 0, 0, 0])

    # Weights (Bryson Rule)
    x_tol = np.array([3.0, 3.0, 10.0])     # position tolerance (design choice)
    v_tol = np.array([0.05, 0.05, 0.05])   # velocity tolerance (design choice)
    u_max = 0.002

    Q_mpc = np.diag(np.concatenate([1/x_tol**2, 1/v_tol**2]))
    R_mpc = np.eye(3) / u_max**2

    P_mpc = scipy.linalg.solve_discrete_are(Ad_mpc, Bd_mpc, Q_mpc, R_mpc)   
    P_mpc = 0.5 * (P_mpc + P_mpc.T)

    # Constraints
    D_u_max = 30*u_max / 25
    R_ko = 20.0

    # Prediction Horizon
    Hp = 10

    # Slack variable
    epsilon = 1
    rho = 1e3
    rho_v = 500

    # mpc
    mpc = LinearMpc(Ad_mpc, Bd_mpc, x_ref, Q_mpc, R_mpc, P_mpc, u_max, D_u_max, R_ko, Hp,rho,rho_v,dt_mpc)


    # plant NL 
    plantNL = NonLinearPlant(a,dt,J2,Re,i,e,omega,RAAN,theta,mu,x0)

    u0 = mpc.compute_control(x0, np.zeros(3))
    print(type(u0), mpc.prob.status, getattr(mpc, "u_opt", None))

    # Simulation
    N = 3000  # Number of time steps
    trajectory_KF, trajectory_true, measures, P_v, u_total_v = run_simulation_NL(plantNL, noise, sensor, kf, mpc, x0, N, u_prev, mult_rate)

    error = trajectory_true - trajectory_KF
    sigma = np.sqrt(np.diagonal(P_v, axis1=1, axis2=2))     # Variance from matrix pile to matrix
    frac = (np.abs(error) < 3*sigma).mean(axis=0)           # Under 3*sigma

    frac_v.append(frac)

# %%

t = np.arange(N + 1) * dt

frac_v = np.array(frac_v)        # shape (n_runs, 6): one row per run
frac_mean = frac_v.mean(axis=0)  # average over runs, one value per state
frac_min  = frac_v.min(axis=0)   # worst run per state

"Q_mis"
X = trajectory_true
U = u_total_v
W = X[1:] - (X[:-1] @ Ad.T + U[:-1] @ Bd.T)   # one-step model residual
Q_emp = np.cov(W.T)                           # includes injected process noise
Q_mis = Q_emp - noise.Q
print(Q_mis)


"Compare the results"

print("\nFraction inside 3_sigma mean:", frac_mean)
print("Fraction inside 3_sigma min:", frac_min)
assert np.all(frac_mean > 0.99)                              # Fail-safe


dv = np.sum(np.linalg.norm(u_total_v, axis=1)) * dt
print("Total Δv [m/s]:", dv)

"Plot the results for visualization"

plt.plot(trajectory_KF[:, 0], trajectory_KF[:, 1])
plt.plot(trajectory_true[:, 0], trajectory_true[:, 1])
plt.gca().add_patch(plt.Circle((0, 0), 20, color='r'))
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.grid()
plt.show()


plt.plot(t, u_total_v[:, 0])
plt.plot(t, u_total_v[:, 1])
plt.plot(t, u_total_v[:, 2])
plt.xlabel("t [s]")
plt.ylabel("u [m/s^2]")
plt.ylim(np.min(np.min(u_total_v)), np.max(np.max(u_total_v)))
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

plt.plot(t, sensor.sigma_p_v)
plt.ylim(0, np.max(sensor.sigma_p_v))
plt.hlines(sensor.sigma_p0,0,N,"r",'--')
plt.xlabel("t [s]")
plt.ylabel("sigma_p [-]")
plt.grid()
plt.show()





# %%

