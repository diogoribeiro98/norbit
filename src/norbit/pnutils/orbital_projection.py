import numpy as np
from scipy import interpolate

from ..vector import vec3, dot, norm
from ..physical_units import units
from .PNclass import nPNsolver

from .orbital_elements import get_apocenter_unit_vectors
from .orbital_elements import get_apocenter_position_and_velocity
from .observer_tetrad  import get_observer_tetrad
from .kepler_period    import kepler_period

class _metric:

    def __init__(self,A,B,D):
        self.A = A
        self.B = B
        self.D = D


class _output:

    def __init__(self,time,alpha,beta,xx,yy,zz,vxx,vyy,vzz,vrs):

        self.tmin = time[0]
        self.tmax = time[-1]

        self.RA  = interpolate.interp1d(time,  alpha)
        self.DEC = interpolate.interp1d(time,  beta )
        self.x   = interpolate.interp1d(time,  xx   )
        self.y   = interpolate.interp1d(time,  yy   )
        self.z   = interpolate.interp1d(time,  zz   )
        self.vx  = interpolate.interp1d(time,  vxx  )
        self.vy  = interpolate.interp1d(time,  vyy  )
        self.vz  = interpolate.interp1d(time,  vzz  )
        self.vrs = interpolate.interp1d(time,  vrs  )


def redshift_factor(r_vec,v_vec,metric):

    r  = norm(r_vec)
    v  = norm(v_vec)

    nr = r_vec/norm(r_vec) 

    return -metric.A(r) + v**2*metric.B(r) + dot(nr,v_vec)**2*metric.D(r)

def redshift_prefactor(r_emmiter,r_observer, v_emmiter, v_observer, metric):
        a1 = redshift_factor(r_observer,v_observer,metric)
        a2 = redshift_factor(r_emmiter ,v_emmiter ,metric)
        
        return np.sqrt(a1/a2)


def get_sky_projection(
        #Orbital elements
        Omega, inc, omega, a, e, 
        #Scale parameters and distance
        m  = units.Rg, 
        R0 = units.R0/units.kiloparsec,
        #GR correction parameters
        orbit_pncor = False,
        light_pncor = False,
        pn_coefficients_1st_order = np.array([-1.0, -2.0, 3.0, 2.0]),
        pn_coefficients_2nd_order = np.array([ 2.0,  0.0, 2.0, 4.0]),
        #Integration values
        tol  = 1e-10,
        tmax = None,
        r_transform = None,
        time_resolution = 1,
        metric = None,
        v_observer = 0.0
):
    """Returns the interpolating function for a given orbit as a function of the observer's time in arcseconds

    Args:
        Omega (float): Longitude of the ascending node in radians
        inc   (float): Inclination in radians
        omega (float): __ in radians 
        a     (float): semi-major axis in Astronomical units
        e     (float): eccentricity (must be between 0 and 1)
        m  (float, optional): Gravitational radius GM/c^2 in meters. Defaults to that of SgrA*.
        R0 (float, optional): Distance of observer. Defaults to galactic center distance in kpc.
        orbit_pncor (bool, optional): If True, integrates the orbits with the 1PN order corrections. Defaults to False.
        light_pncor (bool, optional): If True calculates the effects of light bending on the observed position. Defaults to False.
        pn_coefficients_1st_order (np.array, optional): 1st order PN coefficients. Defaults to np.array([-1,-2, 3, 2]).
        pn_coefficients_2nd_order (np.array, optional): 2nd order PN coefficients. Defaults to np.array([ 2, 0, 2, 4]).
        tol (_type_, optional): Integration tolerance. Defaults to 1e-8.
        tmax (_type_, optional): Maximum integration time in years. Defaults to None, in which case one keplerian period is considered.
        r_transform (lambda, optional): lambda function to be applied to the radial coordinate. Defaults to None.
        time_resolution (_type_, optional): integration resolution in days. Defaults to 1 
    Returns:
        _type_: fx,fy,fz,tmax
    """
    # Observer tetrad and position
    # Note: to match observational conventions, the observer is along 
    #       negative part of the z-axis
    nr,nb,na = get_observer_tetrad( theta_g_deg = 0 , phi_g_deg=180.0)
  
    #Transform quantitites to dimensionless quantities
    a   *= units.astronomical_unit/m 
    R0  *= units.kiloparsec/m

    #Get the position and velocity vector for the apocenter position
    nr_apo, nv_apo          = get_apocenter_unit_vectors(Omega,inc,omega)
    r_apo_norm, v_apo_norm  = get_apocenter_position_and_velocity(a,e)
    
    if r_transform != None:
        r_observer = -(r_transform(R0))*nr 
        r_apo_vec = r_transform(r_apo_norm)*nr_apo
        v_apo_vec = v_apo_norm*nv_apo
    else:
        r_observer = -R0*nr 
        r_apo_vec = r_apo_norm*nr_apo
        v_apo_vec = v_apo_norm*nv_apo

    if tmax == None:
        tmax = kepler_period(a)
    else:
        tmax = tmax*units.year/(m/units.c)

    #Define integral problem
    ode = nPNsolver(
        initial_position= r_apo_vec.values,
        initial_velocity= v_apo_vec.values,
        pncor= orbit_pncor, 
        pn_coefficients_1st_order=pn_coefficients_1st_order,
        pn_coefficients_2nd_order=pn_coefficients_2nd_order,
        tol = tol)

    #Time resolution for evaluations
    dt_eval = time_resolution*units.day/(m/units.c)
    solution = ode.integrate(tf = tmax,dt_eval=dt_eval)

    #Retrieve the data
    time       = []
    alpha,beta = [], []
    xx,yy,zz      = [],[],[]
    vxx,vyy,vzz   = [],[],[]
    vrs = []

    for itt in np.arange(0,len(solution.t)):

        #Get position and velocity
        t  = solution.t[itt]
        x  = solution.y[0, itt]
        y  = solution.y[1, itt]
        z  = solution.y[2, itt]
        vx = solution.y[3, itt]
        vy = solution.y[4, itt]
        vz = solution.y[5, itt]

        #Get light reception angle and corresponding time delay
        deltat, light_vec = ode.deflection_position( vec3([x,y,z]) ,r_observer , pncor=light_pncor)

        #Get corrected velocity (with redshift)
        v_redshift = -(redshift_prefactor(vec3([x,y,z]),r_observer, vec3([vx,vy,vz]), vec3([0,0,v_observer*1000/units.c]), metric)*ode.dtdt0(vec3([x,y,z]),r_observer,vec3([vx,vy,vz]),vec3([0,0,v_observer*1000/units.c]))-1)

        #Append to lists
        time    .append(t+deltat)    
        alpha   .append( dot(light_vec, na)/dot(light_vec, nr) * units.rad_to_as)
        beta    .append( dot(light_vec, nb)/dot(light_vec, nr) * units.rad_to_as)
        xx      .append(x)       
        yy      .append(y)       
        zz      .append(z)       
        vxx     .append(vx)
        vyy     .append(vy)
        vzz     .append(vz)
        vrs     .append(v_redshift)

    #Convert time to years and distances to Astronomical units
    time =  np.array(time)
    time -= time[0] 
    time *= (m/units.c)/units.year

    xx = np.array(xx)*m/units.astronomical_unit 
    yy = np.array(yy)*m/units.astronomical_unit 
    zz = np.array(zz)*m/units.astronomical_unit 

    vxx = np.array(vxx)*units.c/1000.0
    vyy = np.array(vyy)*units.c/1000.0
    vzz = np.array(vzz)*units.c/1000.0
    vrs = np.array(vrs)*units.c/1000.0

    #Return a class with functions
    return _output(time,alpha,beta,xx,yy,zz,vxx,vyy,vzz,vrs)
