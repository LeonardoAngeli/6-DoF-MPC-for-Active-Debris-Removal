import numpy as np
import matplotlib.pyplot as plt

from dynamics.cw_analytical import CWDynamicsAnalytical
from dynamics.ss_equations import SchweighartSedwickDynamics
from dynamics.orbit_propagator import OrbitPropagator


"Test the CW dynamics and orbit propagator for the free motion case (no control input) using both the analytical and numerical solutions."



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

u = np.zeros(3)

# Define the mean motion and time step
mu = 3.986004418e14    # m^3/s^2, parametro gravitazionale terrestre (WGS84)
Re = 6378137.0          # m, raggio equatoriale terrestre
J2 = 1.08263e-3         # adimensionale
h = 800e3               # m, quota del target (scegli tu, 800 km è plausibile per un SL-16)
Rref = Re + h           # m
i = np.radians(71)      # inclinazione target, in RADIANTI (es. 71° è tipico per molti SL-16)


n = np.sqrt(mu / Rref**3)
dt = 1.0

dynamics = SchweighartSedwickDynamics(n, dt, 0, 6378137, 6378137 + 800e3, np.radians(71))
propagator = OrbitPropagator(dynamics)
analytical_dynamics = CWDynamicsAnalytical(n)



"Numerical propagation using the orbit propagator"

N = 20000  # Number of time steps

x = x0.copy()
trajectory = [x.copy()]

for k in range(N):
    x = propagator.propagate(x, u)
    trajectory.append(x.copy())

trajectory = np.array(trajectory)

x_numeric = trajectory[-1]



"Analytical propagation using the CW equations"

t_final = N * dt

x_analytical = analytical_dynamics.propagate(t_final, x0[:3], x0[3:])


"Compare the numerical and analytical results"

error = x_numeric - x_analytical

print("Numerical:", x_numeric)
print("Analytical:", x_analytical)
print("Error:", error)
print("Max error:", np.max(np.abs(error)))


"Plot the results for visualization"

plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.grid()
plt.show()



