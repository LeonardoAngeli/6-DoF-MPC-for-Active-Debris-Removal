import numpy as np
import matplotlib.pyplot as plt


from dynamics.ss_equations import SchweighartSedwickDynamics
from dynamics.orbit_propagator import OrbitPropagator
from sim.simulation import run_simulation
from sim.model_noise import ModelNoise
from sim.sensor_noise import SensorNoise
from estimation.kalman_filter import KalmanFilter


"Test the Kalman filter for the case u=0"

# Initialize the continuous-time dynamics and orbit propagator
x0 = np.array([
    100,
    0.0,           
    0.0,            
    0.0,
    0.0,
    0.0
])

# Note:
#  - x: radial directon b/w the earth and the target satellite
#  - y: along-track direction, perpendicular to the radial direction
#  - z: cross-track direction, perpendicular to the orbital plane

# Input Accelerations
u = np.zeros(3) + np.array([1e-3,0,0])

# Define the mean motion and time step
mu = 3.986004418e14     # m^3/s^2
Re = 6378137.0          # m, Earth Radius
J2 = 1.08263e-3         # adimensional
h = 800e3               # m, Altitude
Rref = Re + h           # m
i = np.radians(71)      # Target Inclination


n = np.sqrt(mu / Rref**3)
dt = 1.0

"KF and true numerical propagation"

## Setting up the structure for simulation ##
seed = 1
rng = np.random.default_rng(seed)

# Plant
dynamics = SchweighartSedwickDynamics(n, dt, J2, Re, Re + h, np.radians(71))
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
x0_kf = np.array([
    60,
    0.1,           
    -0.3,            
    0.4,
    0.08,
    1
])

P0 = np.diag([1000**2]*3 + [30**2]*3)   # σ_pos0 = 5 m, σ_vel0 = 0.5 m/s
kf = KalmanFilter(Ad, Bd, C, noise.Q, sensor.R, x0_kf, P0)

# Simulation
N = 2000  # Number of time steps
trajectory_KF, trajectory_true, measures, P_v = run_simulation(plant, noise, sensor, kf, x0, N, u)






"Compare the results"

error = trajectory_true - trajectory_KF
sigma = np.sqrt(np.diagonal(P_v, axis1=1, axis2=2))     # Variance from matrix pile to matrix
frac = (np.abs(error) < 3*sigma).mean(axis=0)           # Under 3*sigma

print("Fraction inside 3_sigma:", frac)


"Plot the results for visualization"

plt.plot(trajectory_KF[:, 0], trajectory_KF[:, 1])
plt.plot(trajectory_true[:, 0], trajectory_true[:, 1])
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.grid()
plt.show()


t = np.arange(N + 1) * dt
plt.plot(t, error[:, 0], label="error x")
plt.plot(t, 3*sigma[:, 0], "r--", label="+3σ")
plt.plot(t, -3*sigma[:, 0], "r--")
plt.xlabel("t [s]")
plt.ylabel("error x [m]")
plt.legend()
plt.grid()
plt.show()
