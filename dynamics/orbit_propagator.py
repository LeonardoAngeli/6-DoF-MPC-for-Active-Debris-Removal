import numpy as np

class OrbitPropagator:
    def __init__(self, dynamics):
        self.dynamics = dynamics

    def propagate(self, x, u):
        Ad, Bd = self.dynamics.get_discrete_matrices()

        return Ad @ x + Bd @ u

    