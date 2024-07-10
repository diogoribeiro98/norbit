import numpy as np
from ..vector import vec3, cross

def get_angular_momentum_vector(Omega,inc):
    """Returns the angular momentum vector associated with the specified orbital elements

    Args:
        Omega (float): Longitude of the ascending node
        inc   (float): Inclination

    Returns:
        vec3: Angular momentum vector
    """
    return vec3([ np.sin(inc)*np.cos(Omega) , -np.sin(inc)*np.sin(Omega) , -np.cos(inc)  ])


def get_apocenter_position_and_velocity(a, e):
    """Returns the apocenter distance and associated velocity for the specified orbital elements
    
    Args:
        a (float): Semi-major axis in dimensionless units 
        e (float): Eccentricity (must be between 0 and 1)

    Returns:
        tuple(float,float):
        tuple containing

        - **r_apo** (float): apocenter distance
        - **v_apo** (float): velocity at apocenter

    """
    
    r_apo  = a*(1+e) 
    v_apo  = np.sqrt(2/r_apo - 1/a) 
    
    return r_apo, v_apo

def get_pericenter_position_and_velocity(a,e): 
    """Returns the pericenter distance and associated velocity for the specified orbital elements

    Args:
        a (float): Semi-major axis in dimensionless units 
        e (float): Eccentricity (must be between 0 and 1)

    Returns:
        tuple(float,float):
        tuple containing

        - **r_peri** (float): pericenter distance
        - **v_peri** (float): velocity at pericenter

    """
    
    r_peri  = a*(1-e) 
    v_peri  = np.sqrt(2/r_peri - 1/a) 
    
    return r_peri, v_peri


def get_apocenter_unit_vectors(Omega, inc, omega):
    """Returns the apocenter unit vector and and associated velocity unit vector for the specified orbital elements

    Args:
        Omega (float): Longitude of the ascending node in radians
        inc (float): Inclination in radians
        omega (float): Argument of periastron in radians

    Returns:
        tuple(vec3,vec3):
        tuple containing

        - **nr_apo** (vec3): unit vector in the apoceter direction
        - **nv_apo** (vec3): unit vector in the apoceter's velocity direction
    """
 
    #Angular momentum unit vector
    L_vec = get_angular_momentum_vector(Omega,inc)
    
    #Line of ascending nodes unit vector
    anode = vec3([np.sin(Omega),np.cos(Omega),0])

    #Radial direction
    nr_peri = anode.rotate_along(L_vec,omega)
    nr_apo = (-1)*nr_peri
    
    #Velocity direction
    nv_apo = cross(L_vec,nr_apo)

    return nr_apo, nv_apo


def get_pericenter_unit_vectors(Omega, inc, omega):
    """Returns the pericenter unit vector and and associated velocity unit vector for the specified orbital elements

    Args:
        Omega (float): Longitude of the ascending node in radians
        inc (float): Inclination in radians
        omega (float): Argument of periastron in radians

    Returns:
        tuple(vec3,vec3):
        tuple containing

        - **nr_peri** (vec3): unit vector in the pericenter direction
        - **nv_peri** (vec3): unit vector in the pericenter's velocity direction
    """
  

    #Angular momentum unit vector
    L_vec = get_angular_momentum_vector(Omega,inc)
    
    #Line of ascending nodes unit vector
    anode = vec3([np.sin(Omega),np.cos(Omega),0])

    #Radial direction
    nr_peri = anode.rotate_along(L_vec,omega)
    
    #Velocity direction
    nv_apo = cross(L_vec,nr_peri)

    return nr_peri, nv_apo

