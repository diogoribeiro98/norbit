import numpy as np

def kepler_period(a):
    """ Returns the kepler period for a given semi-major axis in dimensionless units.
        To convert the result to seconds muliply by GM/c^3 where M is the mass of the
        central object.

    Args:
        a (float): semi-major axis um gravitational radii.

    Returns:
        float: orbital period in dimensionless units
    """
    return 2*np.pi*np.sqrt(a**3)