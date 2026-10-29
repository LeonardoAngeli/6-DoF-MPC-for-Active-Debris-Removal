import numpy as np
import time

# For the Linear Plant
def run_simulation(plant, noise, sensor, kf, x0_true, N, u):
   
    # Initialazing 
    x_true = x0_true.copy()
    trajectory_KF = [kf.x.copy()]
    trajectory_true = [x_true.copy()]
    measures = []
    P_v = [kf.P.copy()]

    # Simulation
    for k in range(N):
        # True value of x + Noise
        x_true = plant.propagate(x_true, u) + noise.sample()

        # Sensors Data
        y = sensor.measure(x_true)

        # KF
        kf.predict(u)
        kf.update(y)

        # Building arrays
        trajectory_KF.append(kf.x.copy())
        trajectory_true.append(x_true.copy())
        measures.append(y.copy())
        P_v.append(kf.P.copy())

    measures = np.array(measures)
    trajectory_KF = np.array(trajectory_KF)
    trajectory_true = np.array(trajectory_true)
    P_v = np.array(P_v)

    return trajectory_KF, trajectory_true, measures, P_v



# For the NL plant
def run_simulation_NL(plant, noise, sensor, kf, mpc, x0_true, N, u_prev, mult_rate):
    # Closed-loop simulation: nonlinear plant, LIDAR sensor, KF and multi-rate MPC.
    # Plant, sensor and KF run every dt; the MPC is solved every mult_rate steps
    # and its command is held constant in between (ZOH over dt_mpc).
    # Output alignment: u_total_v[k] is the command applied from x[k] to x[k+1].

    # MPC log, one per solve
    mpc_log = {"k": [], "status": [], "iters": [], "t_solve": [], "eps": [], "eps_v": []}

    # Initializing
    x_true = x0_true.copy()
    trajectory_KF = [kf.x.copy()]
    trajectory_true = [x_true.copy()]
    measures = []
    P_v = [kf.P.copy()]

    # First MPC solve before the loop
    t0 = time.perf_counter()
    u_opt = mpc.compute_control(kf.x.copy(), u_prev)
    mpc_log["t_solve"].append(time.perf_counter() - t0)
    mpc_log["k"].append(0)
    mpc_log["status"].append(mpc.prob.status)
    mpc_log["iters"].append(mpc.prob.solver_stats.num_iters)
    mpc_log["eps"].append(float(mpc.epsilon.value[0]))      # keep-out
    mpc_log["eps_v"].append(float(mpc.epsilon_v.value[0]))  # speed-limit

    u_total_v = [u_opt.copy()]
    mult_rate_count = 1

    # Simulation
    for k in range(N):
        # True state from the nonlinear plant, with process disturbance
        x_true = plant.propagateNL(u_opt + noise.sample_accel())

        # Sensor data
        y = sensor.measure(x_true)

        # KF
        kf.predict(u_opt)
        kf.update(y)

        # Building arrays
        trajectory_KF.append(kf.x.copy())
        trajectory_true.append(x_true.copy())
        measures.append(y.copy())
        P_v.append(kf.P.copy())

        # MPC update every mult_rate steps, using the current estimate
        if mult_rate_count == mult_rate:
            t0 = time.perf_counter()
            u_opt = mpc.compute_control(kf.x.copy(), u_opt)
            mpc_log["t_solve"].append(time.perf_counter() - t0)
            mpc_log["k"].append(k + 1)
            mpc_log["status"].append(mpc.prob.status)
            mpc_log["iters"].append(mpc.prob.solver_stats.num_iters)
            mpc_log["eps"].append(float(mpc.epsilon.value[0]))
            mpc_log["eps_v"].append(float(mpc.epsilon_v.value[0]))
            mult_rate_count = 0
        mult_rate_count += 1

        u_total_v.append(u_opt.copy())

    measures = np.array(measures)
    trajectory_KF = np.array(trajectory_KF)
    trajectory_true = np.array(trajectory_true)
    P_v = np.array(P_v)
    u_total_v = np.array(u_total_v)
    mpc_log = {key: np.array(val) for key, val in mpc_log.items()}

    return trajectory_KF, trajectory_true, measures, P_v, u_total_v, mpc_log


        








            


        
        
    