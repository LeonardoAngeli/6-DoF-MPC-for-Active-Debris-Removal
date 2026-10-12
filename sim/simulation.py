import numpy as np

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
def run_simulation_NL(plant, noise, sensor, kf, x0_true, N, u):
    # Initialazing 
        x_true = x0_true.copy()
        trajectory_KF = [kf.x.copy()]
        trajectory_true = [x_true.copy()]
        measures = []
        P_v = [kf.P.copy()]
    
        # Simulation
        for k in range(N):
            # True value of x from NL
            x_true = plant.propagateNL(u)
    
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



        








            


        
        
    