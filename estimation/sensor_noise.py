import numpy as np


class SensorNoise:
    def __init__(self, sigma_p,C, rng):


        ## Sensor Noise ## 

        # Noise Covariance due to sensor precision
        Rs = (sigma_p**2) * np.eye(C.shape[0])    # for the filter
        
        # Noise Covariance
        self.R = Rs
        self.C = C
        self.rng = rng

        # Eigenvalues and Eigenvectors
        mu, D = np.linalg.eigh(self.R)
        mu = mu * (mu > 0)
        self.L = D * np.sqrt(mu)
        

    def measure(self, x):

        ## Get the sensor reading y ##

        # Random unit vector with mean = 0, variance = 1
        r = self.rng.standard_normal(self.R.shape[0])
        
        # Noise
        v = self.L @ r

        # Sensor reading
        y = self.C @ x + v

        return y
    
