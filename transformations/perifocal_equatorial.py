import numpy as np


def perifocal_to_equatorial(r_pf, v_pf, RAAN, omega, i):
    
    RAAN, omega, i = np.deg2rad([RAAN, omega, i])
    cO, sO = np.cos(RAAN), np.sin(RAAN)
    cw, sw = np.cos(omega), np.sin(omega)
    ci, si = np.cos(i), np.sin(i)

    # Q: ECI -> perifocal 
    Q = np.array([
        [ cO*cw - sO*sw*ci,  sO*cw + cO*sw*ci, si*sw],
        [-cO*sw - sO*cw*ci, -sO*sw + cO*cw*ci, si*cw],
        [ sO*si,            -cO*si,            ci   ],
    ])

    # perifocal -> ECI uses the transpose
    return Q.T @ r_pf, Q.T @ v_pf


def equatorial_to_perifocal(r_e, v_e, RAAN, omega, i):
    
    RAAN, omega, i = np.deg2rad([RAAN, omega, i])
    cO, sO = np.cos(RAAN), np.sin(RAAN)
    cw, sw = np.cos(omega), np.sin(omega)
    ci, si = np.cos(i), np.sin(i)

    Q = np.array([
        [ cO*cw - sO*sw*ci,  sO*cw + cO*sw*ci, si*sw],
        [-cO*sw - sO*cw*ci, -sO*sw + cO*cw*ci, si*cw],
        [ sO*si,            -cO*si,            ci   ],
    ])

    
    return Q @ r_e, Q @ v_e