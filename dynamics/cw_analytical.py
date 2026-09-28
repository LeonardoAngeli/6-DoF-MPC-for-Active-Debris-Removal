import numpy as np

class CWDynamicsAnalytical:
    def __init__(self, n):
        "Analytical solution of CW equations for a given mean motion n and time t"

        self.n = n


    def propagate(self, t, r0, v0):

        n = self.n
          

        # State transition matrices for the CW equations

        Phi_rr = np.array([[4 - 3*np.cos(n*t), 0, 0],
                                [6*(np.sin(n*t) - n*t), 1, 0],
                                [0, 0, np.cos(n*t)]])

        Phi_rv = np.array([[1/n*np.sin(n*t), 2/n*(1 - np.cos(n*t)), 0],
                                [-2/n*(1 - np.cos(n*t)), 1/n*(4*np.sin(n*t) - 3*n*t), 0],
                                [0, 0, 1/n*np.sin(n*t)]])

        Phi_vr = np.array([[3*n*np.sin(n*t), 0, 0],
                                [6*n*(np.cos(n*t) - 1), 0, 0],
                                [0, 0, -n*np.sin(n*t)]])

        Phi_vv = np.array([[np.cos(n*t), 2*np.sin(n*t), 0],
                                [-2*np.sin(n*t), 4*np.cos(n*t) - 3, 0],
                                [0, 0, np.cos(n*t)]])

    
        # Recover the state at time t using the analytical solution
        # Note: r_t and v_t are relative position and velocity vectors at time t, respectively.

        r_t = Phi_rr @ r0 + Phi_rv @ v0
        v_t = Phi_vr @ r0 + Phi_vv @ v0

        # Concatenate the relative position and velocity vectors to form the state vector at time t
        x_t = np.concatenate((r_t, v_t), axis=0)

        return x_t







