import numpy as np


# Transformation from ECI to the LVLH of the Target
def matrix_eci_to_LVLH(r_T, v_T):

    # X_coord LVLH
    x = r_T / np.linalg.norm(r_T)

    # Z_coord LVLH (Perpendicular to orbital plane)
    z = (np.cross(r_T,v_T)) / np.linalg.norm(np.cross(r_T,v_T))

    # Y_coord LVLH
    y = np.cross(z,x)


    Q_L = np.array([x, y, z])

    return Q_L

