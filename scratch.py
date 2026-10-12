# %%
%load_ext autoreload
%autoreload 2

import numpy as np
from dynamics.nonlinear_plant import NonLinearPlant
# %%
mu = 3.986004418e14; Re = 6378137.0
p = NonLinearPlant(Re+800e3, 1.0, 1.08263e-3, Re, 71.0, 0.0, 0.0, 0.0, 0.0, mu,
                   np.array([0., -2000., 0., 0., 0., 0.]))

# %%
p._to_relative(p.y)   # deve restituire [0, -2000, 0, 0, 0, 0]
# %%
