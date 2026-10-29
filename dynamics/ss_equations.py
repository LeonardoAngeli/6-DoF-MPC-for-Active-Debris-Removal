import numpy as np
from scipy.signal import cont2discrete


class SchweighartSedwickDynamics:
    # Linearized relative dynamics with mean J2 effects (Schweighart-Sedwick),
    # LVLH frame, state x = [x, y, z, vx, vy, vz], input u = thrust acceleration.

    def __init__(self, n, dt, J2, Re, Rref, i):
        # n: mean motion [rad/s], 
        # dt: time step [s], 
        # Rref: reference orbit radius [m],
        # i: inclination [rad]

        self.n = n
        self.dt = dt
        self.J2 = J2
        self.Re = Re
        self.Rref = Rref
        self.i = i

        # J2 correction coefficients
        c = np.sqrt(1 + (3 * J2 * Re**2) / (8 * Rref**2) * (1 + 3 * np.cos(2 * i)))
        k = n * c + (3 * n * J2 * Re**2) / (2 * Rref**2) * np.cos(i)**2

        # Continuous-time state matrix
        self.Ac = np.array([[0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1],
                            [(5 * c**2 - 2) * n**2, 0, 0, 0, 2 * n * c, 0],
                            [0, 0, 0, -2 * n * c, 0, 0],
                            [0, 0, -k**2, 0, 0, 0]])

        # Continuous-time input matrix
        self.Bc = np.vstack([np.zeros((3, 3)), np.eye(3)])

        # Zero-order-hold discretization
        self.Ad, self.Bd, _, _, _ = cont2discrete(
            (self.Ac, self.Bc, np.eye(6), np.zeros((6, 3))), self.dt)

    def get_discrete_matrices(self):
        # Discrete-time matrices (Ad, Bd)
        return self.Ad, self.Bd

    def get_continuous_matrices(self):
        # Continuous-time matrices (Ac, Bc)
        return self.Ac, self.Bc