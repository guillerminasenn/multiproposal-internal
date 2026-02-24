# ESS = MESS with M = 1

# algorithms/ess.py
import numpy as np
from .mess import mess_step
# export PYTHONPATH=src

def ess_step(x, problem, rng):
    """Perform an ESS step.
    Parameters
    ----------
    x : np.ndarray
        Current state.
    problem : multiproposal.problems.base.Problem
        Problem instance.
    rng : np.random.Generator
        Random number generator.
    """
    
    # Run MESS with M = 1
    return mess_step(x, problem, rng)
