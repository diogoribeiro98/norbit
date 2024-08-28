import numpy as np

def get_observer_tetrad(theta_g_deg , phi_g_deg):
    """ 
    Returns Cartesian frame associated with an observer in the direction given by the polar coordinates theta and phi. 
    The radial component points to the origin while na and nb point in the tangential theta and phi directions respectively.

    Example: 
    
    For theta = 0 phi = 0, the vectors are

    nr = [0,0,-1]  nb = [0,1,0]  na = [1,0,0]
    
    while for theta = 90 phi = 0 the vectors are
    
    nr = [0,-1,0]  nb = [0,0,-1]  na = [1,0,0]

    
    Args:
        theta_g_deg float: theta angle in degrees
        phi_g_deg   float: phi angle in degrees

    Returns:
        nr,nb,na: returns three vec3 vectors defining the observer frame 
    """

    theta_g =  np.deg2rad(theta_g_deg) 
    phi_g   =  np.deg2rad(phi_g_deg)
    
    #Observer basis vectors
    nr = np.array([ -np.cos(theta_g)*np.sin(phi_g)   ,
                -np.sin(theta_g)                 ,
                -np.cos(theta_g)*np.cos(phi_g)   ])

    nb = np.array([ -np.sin(theta_g)*np.sin(phi_g)   ,
                 np.cos(theta_g)                  ,
                -np.sin(theta_g)*np.cos(phi_g)   ])

    na = np.array([  np.cos(phi_g)  ,
                 0              ,
                -np.sin(phi_g)   ])

    return nr, nb, na
