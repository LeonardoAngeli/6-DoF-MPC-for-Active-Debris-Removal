import numpy as np
from scipy.signal import cont2discrete


class CWDynamics:
    def __init__(self, n, dt):
        " Initialize the continuous-time dynamics matrices "

        self.n = n
        self.dt = dt
        
        # Ac is the continuous-time dynamics matrix for the CW equations
        self.Ac = np.array([[0, 0, 0, 1, 0,0], 
                            [0, 0, 0, 0, 1, 0], 
                            [0, 0, 0, 0, 0, 1], 
                            [3*(n**2), 0, 0, 0, 2*n, 0], 
                            [0, 0, 0, -2*n, 0, 0], 
                            [0, 0, -(n**2), 0, 0, 0]])  

        # Bc is the continuous-time input matrix for the CW equations
        self.Bc = np.array([[0, 0, 0], 
                            [0, 0, 0], 
                            [0, 0, 0], 
                            [1, 0, 0], 
                            [0, 1, 0], 
                            [0, 0, 1]])


        # Discretize the continuous-time dynamics matrices using zero-order hold
        self.Ad, self.Bd, _, _, _ = cont2discrete((self.Ac, self.Bc, np.eye(6), np.zeros((6, 3))), self.dt)

        
    
        

    def get_discrete_matrices(self):
        " Return the discrete-time dynamics matrices Ad and Bd "
        
        return self.Ad, self.Bd





# ==========================================
# Rapid test area
# ==========================================
if __name__ == "__main__":
    print("Rapid test of CWDynamics\nTesting the continuous-time dynamics matrices and their discretization.\n")
    CWDynamics_instance = CWDynamics(n=2, dt=0.1)
    print(CWDynamics_instance.Ac)
    Ad, Bd = CWDynamics_instance.get_discrete_matrices()
    print("Discrete-time dynamics matrices:\n", Ad,"\n\n", Bd)

    