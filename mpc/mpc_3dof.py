import numpy as np
import cvxpy as cp


class LinearMpc:
    # Linear MPC for the translational rendezvous.
    # The problem is built once with cvxpy Parameters and re-solved at each step.

    def __init__(self, Ad, Bd, x_ref, Q_mpc, R_mpc, P_mpc, u_max, D_u_max, R_ko, Hp, rho, rho_v, dt):

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
        self.dt = dt

        # cvxpy variables and parameters
        self.x = cp.Variable((6, Hp + 1))
        self.u = cp.Variable((3, Hp))
        self.x0 = cp.Parameter((6,))
        self.u_prev = cp.Parameter((3,))
        self.n = cp.Parameter((3,))

        # Half-space keep out normal vector: line to the reference point, fixed
        # for the whole approach.
        self.n.value = self.x_ref[:3] / np.linalg.norm(self.x_ref[:3])

        # Linear penalties
        self.epsilon = cp.Variable(Hp)      # Keep-out
        self.epsilon_v = cp.Variable(Hp)    # Speed limit

        # Speed limit: the horizon must contain the braking manoeuvre.
        self.v_max = 0.8 * u_max * Hp * dt

        # Terminal cost
        V = cp.quad_form(self.x_ref - self.x[:, Hp], self.P_mpc)

        constraints = [
            self.x[:, 0] == self.x0,
            self.epsilon >= 0,
            self.epsilon_v >= 0,
        ]

        for k in range(Hp):
            # Stage cost
            V += (cp.quad_form(self.x_ref - self.x[:, k], self.Q_mpc)
                  + cp.quad_form(self.u[:, k], self.R_mpc)
                  + self.rho * self.epsilon[k]
                  + self.rho_v * self.epsilon_v[k])

            constraints += [
                # Linear dynamics
                self.x[:, k + 1] == Ad @ self.x[:, k] + Bd @ self.u[:, k],

                # Keep-out half-space 
                self.R_ko - self.n.T @ self.x[:3, k + 1] <= self.epsilon[k],

                # Thrust limit
                cp.norm(self.u[:, k], 2) <= self.u_max,

                # Speed limit
                cp.norm(self.x[3:, k + 1], 2) <= self.v_max + self.epsilon_v[k],
            ]

            # Thrust rate limit (before including 6DoF sim)
            if k >= 1:
                constraints += [cp.norm(self.u[:, k] - self.u[:, k - 1], 2) <= D_u_max]
            else:
                constraints += [cp.norm(self.u[:, 0] - self.u_prev, 2) <= D_u_max]

        self.prob = cp.Problem(cp.Minimize(V), constraints)

    def compute_control(self, x0, u_prev):
        # Solve the MPC from the current state estimate and return the first u

        self.x0.value = x0
        self.u_prev.value = u_prev

        self.prob.solve()

        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC failed: status={self.prob.status}, x0={x0}, u_prev={u_prev}")

        self.u_opt = self.u[:, 0].value.copy()
        return self.u_opt