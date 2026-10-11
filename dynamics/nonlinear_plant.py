import numpy as np

class NonLinearPlant:
    def __init__(self, a, dt, J2, Re, Rref, i, e, omega, RAAN, theta):

        self.a = a
        self.dt = dt
        self.J2 = J2
        self.Re = Re
        self.Rref = Rref
        self.i = i
        self.e = e
        self.omega = omega
        self.theta = theta
        self.RAAN = RAAN

        self.r0 = a * (1 - e**2) / (1 + e*np.sin(np.deg2rad(theta)));
        self.


        pass