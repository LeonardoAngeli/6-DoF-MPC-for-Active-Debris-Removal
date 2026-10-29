import numpy as np


class SensorNoise:
    # LIDAR relative-position sensor, modelled on the Jena-Optronik RVS3000-3D.
    # Range-dependent noise:
    #   sigma_p(r) = sigma_p0 + k_r * r
    #   k_r     = 0.001 / 3   (range dependant noise factor)
    #   sigma_p0 = 0.1 m      (base noise)

    # The true measurement noise uses the true range; the KF covariance R uses the
    # estimated range, since the filter does not know the true state.

    def __init__(self, C, rng):
        self.C = C
        self.rng = rng
        self.n_meas = C.shape[0]

        # Noise parameters
        self.sigma_p0 = 0.1             # m
        self.k_r = 1.0 / 3e3            # -

        self.sigma_p_v = []

    def sigma_p(self, x):
        # Noise standard deviation at x
        r = np.linalg.norm(x[:3])
        return self.sigma_p0 + self.k_r * r

    def get_matrix_R(self, x):
        # Measurement covariance for the KF, evaluated at the estimated state
        return self.sigma_p(x)**2 * np.eye(self.n_meas)

    def measure(self, x):
        # Sensor reading y = C x + v
        sigma = self.sigma_p(x)
        self.sigma_p_v.append(sigma)

        v = sigma * self.rng.standard_normal(self.n_meas)
        return self.C @ x + v