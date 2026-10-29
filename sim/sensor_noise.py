import numpy as np


class SensorNoise:
    def __init__(self,C, rng, x0_kf):


        ## Sensor Noise ## 
        # Jena-Optronik RVS3000-3D
        # Simple model

        # Default
        self.n_meas = C.shape[0]
        self.C = C
        self.rng = rng

        # Sensor variance
        self.sigma_p0 = 0.1

        # Distance to target
        r = np.linalg.norm(x0_kf[:3])
                
        # Total variance sensor 
        sigma_p = self.sigma_p0 + r/3e3

        # Noise Covariance due to sensor precision
        Rs = (sigma_p**2) * np.eye(self.C.shape[0])    # for the filter
                        
        # Noise Covariance
        self.R = Rs

        # Eigenvalues and Eigenvectors
        mu, D = np.linalg.eigh(self.R)
        mu = mu * (mu > 0)
        self.L = D * np.sqrt(mu)

        # Save vector
        self.sigma_p_v = [sigma_p.copy()]


        
        

    def get_matrix_R(self,x):

        # Distance to target
        r = np.linalg.norm(x[:3])
        
        # Total variance sensor 
        sigma_p = self.sigma_p0 + r/3e3
        

        # Noise Covariance due to sensor precision
        Rs = (sigma_p**2) * np.eye(self.C.shape[0])    # for the filter
                
        # Noise Covariance
        self.R = Rs

        # Eigenvalues and Eigenvectors
        mu, D = np.linalg.eigh(self.R)
        mu = mu * (mu > 0)
        self.L = D * np.sqrt(mu)

        # Save vector
        self.sigma_p_v.append(sigma_p.copy())

        return self.R


        
    def measure(self, x):

        ## Get the sensor reading y ##

        # Random unit vector with mean = 0, variance = 1
        r = self.rng.standard_normal(self.R.shape[0])
        
        # Noise
        v = self.L @ r

        # Sensor reading
        y = self.C @ x + v

        return y
    
