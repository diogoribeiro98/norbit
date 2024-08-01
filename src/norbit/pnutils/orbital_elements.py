import numpy as np
from .kepler_period import kepler_period
from ..vector import vec3, cross, norm
from scipy import optimize

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

#
# Solving Kepler's problem
#

#John Machin's method for the inital guess

def solve_cubic(a, c, d):
    
    assert(a > 0 and c > 0)
    
    p = c/a
    q = d/a
    k = np.sqrt( q**2/4 + p**3/27 )
    
    return np.cbrt(-q/2 - k) + np.cbrt(-q/2 + k)

# Machin's starting point for Newton's method
# See johndcook.com/blog/2022/11/01/kepler-newton/
def machin(e, M):
    n = np.sqrt(5 + np.sqrt(16 + 9/e))
    a = n*(e*(n**2 - 1)+1)/6
    c = n*(1-e)
    d = -M
    s = solve_cubic(a, c, d)
    return n*np.arcsin(s)    

def eccentric_anomaly(e, M):
    "Find E such that M = E - e sin E."
   
    assert(0 <= e < 1)
  
    f = lambda E: E - e*np.sin(E) - M 
    fprime = lambda E: 1 - e*np.cos(E) 
    
    #Initial guess
    E = M 

    #Note: if a better guess is needed, use the following line instead
    #E = machin(e, M) 

    E = optimize.newton(f,E,fprime=fprime, maxiter=500, rtol=1e-13)

    return E

def true_anomaly(e,E):
    return 2*np.arctan( np.sqrt((1+e)/(1-e)) * np.tan(E/2))


def get_position_and_velocity_at_t0(
        time_since_periapsis,
        a,e,
        Omega, inc, omega
):
    """Given the time since periapsis in dimensionless units, the orbit's semi-major axis and excentricity, returns the position and velocity vectors at that point 

    Args:
        time_since_periapsis (float): time since periapsis
        a (float): Semi-major axis in dimensionless units 
        e (float): Eccentricity (must be between 0 and 1)
    """
    
    #Mean anomaly
    #Me = 2*np.pi*np.mod(time_since_periapsis/kepler_period(a), 1.0)
    Me = 2*np.pi*time_since_periapsis/kepler_period(a)
    
    #Solve kepler's problem
    E  = eccentric_anomaly(e, Me)
    nu = true_anomaly(e,E)

    #Get perifocal unit vectors
    p, q = get_pericenter_unit_vectors(Omega, inc, omega)

    #Get initial position in the perifocal frame
    snu = np.sin(nu)
    cnu = np.cos(nu)

    rnorm = a*(1-e**2)/(1+e*cnu)
    vnorm = 1/np.sqrt( a*(1-e**2) )
    
    pos = rnorm*( cnu*p + snu*q )
    vel = vnorm*(-snu*p + (e+cnu)*q)
    
    return pos, vel

    