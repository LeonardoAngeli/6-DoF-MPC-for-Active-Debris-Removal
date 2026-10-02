import numpy as np
from scipy.signal import cont2discrete


class SchweighartSedwickDynamics:
    def __init__(self, n, dt, J2, Re, Rref, i):
        " Initialize the continuous-time dynamics matrices "

        self.n = n
        self.dt = dt
        self.J2 = J2
        self.Re = Re
        self.Rref = Rref
        self.i = i


        c = np.sqrt(1 + (3*J2*(Re**2) / (8*Rref**2) * (1 + 3 * np.cos(2*i))))
    
        
        # Ac is the continuous-time dynamics matrix for the SS equations
        self.Ac = np.array([[0, 0, 0, 1, 0,0], 
                            [0, 0, 0, 0, 1, 0], 
                            [0, 0, 0, 0, 0, 1], 
                            [(5*(c**2) - 2)*(n**2), 0, 0, 0, 2*n*c, 0], 
                            [0, 0, 0, -2*n*c, 0, 0], 
                            [0, 0, -(3*c**2 - 2) * n**2, 0, 0, 0]])  

        # Bc is the continuous-time input matrix for the SS equations
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

    def get_continuous_matrices(self):
        " Return the continuous-time dynamics matrices Ac and Bc "
        
        return self.Ac, self.Bc

    







    