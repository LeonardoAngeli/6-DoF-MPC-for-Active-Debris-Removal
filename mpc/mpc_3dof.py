import numpy as np
import cvxpy as cp
from numpy import linalg as la

class LinearMpc:
    def __init__(self, Ad, Bd, x_ref, Q_mpc, R_mpc, P_mpc, u_max, D_u_max, R_ko, Hp,rho,rho_v,dt):

        self.Ad = Ad
        self.Bd = Bd
        self.x_ref = x_ref
        self.Q_mpc = Q_mpc
        self.R_mpc = R_mpc
        self.P_mpc = P_mpc
        self.u_max = u_max
        self.D_u_max = D_u_max
        self.R_ko = R_ko
        self.Hp = Hp
        self.rho = rho
        self.rho_v = rho_v

        # Construct problem
        self.x = cp.Variable((6,Hp+1))
        self.u = cp.Variable((3,Hp))
        self.x0 = cp.Parameter((6,))
        self.u_prev = cp.Parameter((3,))
        self.n = cp.Parameter((3,))

        # Slack variable
        self.epsilon = cp.Variable(Hp)              # Cost of the slack is just linear and not quadratic, so that smaller variations more important

        # Max relative speed
        self.v_max = Hp * dt * u_max * 0.8
        self.epsilon_v = cp.Variable(Hp)


        V = cp.quad_form(self.x_ref - self.x[:,Hp], self.P_mpc)
        constraints = [
            self.x[:, 0] == self.x0,
                       
            # Slack variable
            self.epsilon >= 0,
            self.epsilon_v >= 0

            ]
        


        for k in range(Hp):
            V += cp.quad_form(self.x_ref - self.x[:,k], self.Q_mpc) + cp.quad_form(self.u[:,k], self.R_mpc) + self.rho * (self.epsilon[k]) + self.rho_v * (self.epsilon_v[k])

            constraints += [
                           
                        # Define model of the system
                        self.x[:,k+1] == Ad @ self.x[:,k] + Bd @ self.u[:,k],
            
                        # Set limit of 20m
                        self.R_ko - (self.n.T @ self.x[:3, k+1]) <= self.epsilon[k],      # Using the slack variable I make the constraint more permissive and solve the infesability issue 
            
                        # Max thrust
                        cp.norm(self.u[:, k], 2) <= self.u_max,

                        # Speed limit
                        cp.norm(self.x[3:, k+1],2) <= self.v_max + self.epsilon_v[k] 
            
                        ]

            if k >= 1:
                constraints += [ 
                               cp.norm(self.u[:,k] - self.u[:,k-1]) <= D_u_max
                               ]

            else:
                constraints += [ 
                               cp.norm(self.u[:,0] - self.u_prev) <= D_u_max
                               ]

        objective = cp.Minimize(V)

        # Problem
        self.prob = cp.Problem(objective,constraints)


    def compute_control(self,x0,u_prev):

        self.x0.value = x0
        self.u_prev.value = u_prev
        self.n.value = x0[:3] / np.linalg.norm(x0[:3])

        self.prob.solve()

        # Fail check
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC failed: status={self.prob.status}, x0={x0}, u_prev={u_prev}")
        self.u_opt = self.u[:, 0].value.copy()
        return self.u_opt

        
        










        