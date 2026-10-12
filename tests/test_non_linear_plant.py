# %%
%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt


from dynamics.ss_equations import SchweighartSedwickDynamics
from dynamics.orbit_propagator import OrbitPropagator
from dynamics.nonlinear_plant import NonLinearPlant



# %%

"Compare the SS model dynamics with NLdynamics (w/out Kalman Filter)"

# Initial positions and speed of the Chaser
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

u = np.zeros(3)

# Define the motion and time step
mu = 3.986004418e14    # m^3/s^2, Gravitational parameter Earth
Re = 6378137.0          # m, Earth Radius
J2 = 1.08263e-3         # adimensional
i = 71                  # inclination
e = 0.0004              # eccentricity
omega = 69              # argument of the perigee
RAAN = 20               # RAAN
theta = 0               # anomaly
a = Re + 800e3               # semi-major axis of the orbit

n = np.sqrt(mu / a**3)
dt = 1.0

# SS Model
dynamics = SchweighartSedwickDynamics(n, dt, J2, Re, a, np.deg2rad(i))
propagator = OrbitPropagator(dynamics)

# NL Model
dynamicsNL = NonLinearPlant(a,dt,J2,Re,i,e,omega,RAAN,theta,mu,x0)



"Numerical propagation"

# SS Propagation
N = 20000  # number of time steps

xSS = x0.copy()
trajectorySS = [xSS.copy()]

for k in range(N):
    xSS = propagator.propagate(xSS, u)
    trajectorySS.append(xSS.copy())

trajectorySS = np.array(trajectorySS)
x_final_SS = trajectorySS[-1]

# NL Propagation
xNL = x0.copy()
trajectoryNL = [xNL.copy()]

for k in range(N):
    xNL = dynamicsNL.propagateNL(u)
    trajectoryNL.append(xNL.copy())

trajectoryNL = np.array(trajectoryNL)
x_final_NL = trajectoryNL[-1]


"Results"

error_v = trajectorySS - trajectoryNL
error_max = np.max(np.max(np.abs(error_v)))

"Plot the results for visualization"

plt.plot(trajectoryNL[:, 0], trajectoryNL[:, 1])
plt.plot(trajectorySS[:, 0], trajectorySS[:, 1])
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.grid()
plt.show()


t = np.arange(N + 1) * dt
plt.plot(t, error_v[:, 0], label="error x")
plt.xlabel("t [s]")
plt.ylabel("error x [m]")
plt.legend()
plt.grid()
plt.show()


# %%
