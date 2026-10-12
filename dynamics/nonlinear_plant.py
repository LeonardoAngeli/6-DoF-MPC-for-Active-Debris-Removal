import numpy as np
import scipy as scipy


from transformations.perifocal_equatorial import perifocal_to_equatorial
from transformations.eci_lvlh import matrix_eci_to_LVLH
from scipy import integrate

class NonLinearPlant:
    def __init__(self, a, dt, J2, Re, i, e, omega, RAAN, theta, mu, x0):

        self.a = a
        self.dt = dt
        self.J2 = J2
        self.Re = Re
        self.i = i
        self.e = e
        self.omega = omega
        self.theta = theta
        self.RAAN = RAAN
        self.mu = mu

        # Relative distance and relative speed of the chaser in the LVLH frame
        self.rel_r_C = x0[:3]
        self.rel_v_C = x0[3:]


        # Find r0 and v0 for perifocal form orbital parameters of the target
        theta = np.deg2rad(theta)    

        p = a * (1 - e**2)
        r = p / (1 + e * np.cos(theta))
        r_pf = r * np.array([np.cos(theta), np.sin(theta), 0.0])
        v_pf = np.sqrt(mu / p) * np.array([-np.sin(theta), e + np.cos(theta), 0.0])

        # From perifocal to ECI Target
        self.r0_T, self.v0_T = perifocal_to_equatorial(r_pf, v_pf, RAAN, omega, i)

        # For chaser we want to find its ECI position given its relative position and velocity to target in LVLH
        # Target ECI position and Velocity
        Q_L = matrix_eci_to_LVLH(self.r0_T, self.v0_T)

        # Relative Angular Velocity
        omega_L = self._omega_L(self.r0_T,self.v0_T,Q_L)

        # Chaser ECI position and Velocity
        self.r0_C = self.r0_T + Q_L.T @ self.rel_r_C
        self.v0_C = self.v0_T + Q_L.T @ (self.rel_v_C + np.cross(omega_L, self.rel_r_C))

        # State for ODE
        self.y =np.concatenate((self.r0_T,self.v0_T, self.r0_C, self.v0_C))





    def _a_J2(self,r):
        # J2 Acceleration
        # Note: For J2 we use cartesian coordinates
        xr = r[0]
        yr = r[1]
        zr = r[2]

        rn = np.linalg.norm(r)
        a_J2 = ( (-(3*self.J2 * self.mu * (self.Re**2)) / (2 * rn**5))
                   * np.array([xr*(1-5*zr**2/(rn**2)), yr*(1-5*zr**2/(rn**2)), zr*(3-5*zr**2/(rn**2)) ]) )
        
        return a_J2


    
    def _omega_L(self,r_T,v_T, Q_L):
        # Relative Angular Velocity

        # Norm r Target
        rn_T = np.linalg.norm(r_T)

        # Angular momentum
        h = np.linalg.norm(np.cross(r_T, v_T))
    
        # Relative Angular Velocity
        ph = self._a_J2(r_T) @ Q_L[2]
        omega_L = np.array([rn_T*ph/h, 0, h/rn_T**2])

        return omega_L


    # ODE Definition for both target and chaser
    def fun(self,t, y, u):
        r_T = y[:3]
        v_T = y[3:6]
        r_C = y[6:9]
        v_C = y[9:]
        
        

        # Acceleration: Gravity + J2 Target
        a_T = (- self.mu * r_T / (np.linalg.norm(r_T)**3)) + self._a_J2(r_T)

        # For a_u the Q_L needs to be re-written at every step as LVLH changes
        # a_u
        Q_L = matrix_eci_to_LVLH(r_T,v_T)           
        a_u = Q_L.T @ u

        # Acceleration: Gravity + J2 + a_u Chaser
        a_C = (- self.mu * r_C / (np.linalg.norm(r_C)**3)) + self._a_J2(r_C) + a_u
        
        return np.concatenate((v_T, a_T, v_C, a_C))    
        




    def propagateNL(self, u):
        # For increased precision DOP853 was chosen
        
        sol = integrate.solve_ivp(
            self.fun,
            t_span=(0.0, self.dt),        # (t_start, t_end)
            y0=self.y,
            method='DOP853',
            rtol=1e-12,                   # Relative Tolerance
            atol=1e-6,                    # Absolute Tolerance
            args=(u,),
        )

        self.y = sol.y[:,-1]
        return self._to_relative(self.y)


    def _to_relative(self,y):
        # Return from ECI to LVLH

        Q_L = matrix_eci_to_LVLH(y[:3],y[3:6])
        omega_L = self._omega_L(y[:3],y[3:6],Q_L)

        self.rel_r_C = Q_L @ (y[6:9] - y[:3])
        self.rel_v_C = Q_L @ (y[9:] - y[3:6]) - np.cross(omega_L,self.rel_r_C)

        return np.concatenate((self.rel_r_C,self.rel_v_C))