from dataclasses import dataclass
import numpy as np


@dataclass
class OrbitParams:

    mu: float
    Rref: float
    J2: float
    Re: float
    i:float

    