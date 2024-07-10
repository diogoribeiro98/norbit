import numpy as np

def kepler_period(a):
    """ Returns the kepler period for a given semi-major axis in code units.

    Args:
        a (float): semi-major axis in gravitational radii.

    Returns:
        float: orbital period in dimensionless units.
    """
    return 2*np.pi*np.sqrt(a**3)