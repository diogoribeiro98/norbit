import numpy as np
from ..vector import vec3, cross

def get_angular_momentum_vector(Omega,inc):
    return vec3([ np.sin(inc)*np.cos(Omega) , -np.sin(inc)*np.sin(Omega) , -np.cos(inc)  ])

def get_apocenter_position_and_velocity(a, e):
    r_apo  = a*(1+e) 
    v_apo  = np.sqrt(2/r_apo - 1/a) 
    return r_apo, v_apo

def get_apocenter_unit_vectors(Omega, inc, omega):
 
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