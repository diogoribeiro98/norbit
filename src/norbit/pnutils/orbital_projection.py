import numpy as np
from scipy import interpolate

from ..vector import vec3, dot, norm
from ..physical_units import units
from .PNclass import nPNsolver

from .orbital_elements import get_apocenter_unit_vectors
from .orbital_elements import get_apocenter_position_and_velocity
from .orbital_elements import get_position_and_velocity_at_t0
from .observer_tetrad  import get_observer_tetrad
from .kepler_period    import kepler_period

from .metric import schwarzschild_metric

class _output:
    """Helper output class
    """

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

class _output_fit:
    """Simplified output helper class
    """

    def __init__(self,time,alpha,beta,vrs):

        self.tmin = time[0]
        self.tmax = time[-1]

        self.RA  = interpolate.interp1d(time,  alpha)
        self.DEC = interpolate.interp1d(time,  beta )
        self.vrs = interpolate.interp1d(time,  vrs  )

class orbital_projection():

    def __init__(self):
        return

    def get_sky_projection(self,
        #Orbital elements
        Omega, inc, omega, a, e, 
        #Scale parameters and distance
        m  = units.Rg, 
        R0 = units.R0/units.parsec,
        v_observer = [0., 0., 0.],
        #GR metric
        metric = schwarzschild_metric,
        r_transform = None,
        #Post-newtonian corrections
        orbit_pncor=True,
        light_pncor=True,
        light_travel_time=True,
        gr_redshift=True,
        sr_redshift=True,
        #Integration values
        tmin   = 0.0,
        tmax   = None,
        time_resolution = 1.0
        ):
        """
        Returns the interpolating function for a given orbit as a function of the observer's time. 
        Currently only allows integration from apocenter.
        
        Args:
            Omega   (float): Longitude of the ascending node in radians
            inc     (float): Inclination in radians
            omega   (float): argument of pericenter passage in radians
            a       (float): semimajor axis in Astronomical Units
            e       (float): eccentricity (must be between 0 and 1)
            m       (float, optional): Central body's gravitational radius in meters. Defaults to that of SgrA*.
            R0      (float, optional): Distance between observer and central body. Defaults to GC distance in parsecs.
            metric (norbit.metric, optional): Spacetime metric. Defaults to schwarzschild_metric.
            r_transform (lambda function, optional): Radial coordinate transformation. Defaults to None.
            orbit_pncor (bool, optional): If True, considers the first order Post-newtonian effects in the EOM. Defaults to True.
            light_pncor (bool, optional): If True, considers the first order Post-newtonian effects in the trajectory of photons connecting the orbiting body and the observer. Defaults to False.
            light_travel_time (bool, optional): If False, assumes speed of light to be infinite. Defaults to True.
            gr_redshift (bool, optional): If False, discards the gravitational field effects on the emmited photon's redshift. Defaults to True.
            sr_redshift (bool, optional): If False, discards the special relativity effects from the emitter's motion on the photon's redshift. Defaults to False.
            tol (float, optional): Tolerance for the integrator. Defaults to 1e-10.
            ti (float, optional): Minimum integration time in years. Defaults to 0.0
            tf (float, optional): Maximum integration time in years. If None, uses the period of one single orbital revolution.
            r_transform (lambda function, optional): Radial coordinate transformation as a lambda function.
            time_resolution (float, optional): Time resolution of the integration in days. Defaults to 1.0.
            v_observer (list, optional): Observer's velocity with respect to the central body. Defaults to [0., 0., 0.].

        Returns:
            output: class with interpolated function
        """    
    
        # Observer tetrad and position
        # Note: to match observational conventions, the observer is along negative part of the z-axis
        nr,nb,na = get_observer_tetrad( theta_g_deg = 0 , phi_g_deg=180.0)
    
        #Transform quantitites to dimensionless quantities
        a   *= units.astronomical_unit/m 
        R0  *= units.parsec/m

        #Get the position and velocity vector for the apocenter position
        nr_apo, nv_apo          = get_apocenter_unit_vectors(Omega,inc,omega)
        r_apo_norm, v_apo_norm  = get_apocenter_position_and_velocity(a,e)
        
        #Radial coordinate transformation
        if r_transform != None:
            r_observer = -(r_transform(R0))*nr 
            r_apo_vec = r_transform(r_apo_norm)*nr_apo
            v_apo_vec = v_apo_norm*nv_apo
        else:
            r_observer = -R0*nr 
            r_apo_vec = r_apo_norm*nr_apo
            v_apo_vec = v_apo_norm*nv_apo

        v_obs = vec3(v_observer)*1000/units.c

        #Maximum integration time in code units
        if tmax == None:
            tmax = kepler_period(a)
        else:
            tmax *= units.year/(m/units.c)

        tmin *= units.year/(m/units.c)


        #Define integral problem
        ode = nPNsolver(
            initial_position= r_apo_vec.values,
            initial_velocity= v_apo_vec.values,
            metric=metric)

        #Time resolution for evaluations
        dt_eval = time_resolution*units.day/(m/units.c)
        
        solution = ode.integrate(
            ti=tmin, 
            tf=tmax, 
            dt_eval=dt_eval, 
            pncor=orbit_pncor)

        #Calculate romer delay for the initial point
        deltat_init, _ = ode.deflection_position(ri=r_apo_vec,rf=r_observer ,pncor=light_pncor, light_travel_time=light_travel_time)

        #Retrieve the data
        time        = []
        alpha,beta  = [], []
        xx,yy,zz    = [],[],[]
        vxx,vyy,vzz = [],[],[]
        vrs         = []

        for itt in np.arange(0,len(solution.t)):

            #Get position and velocity
            t  = solution.t[itt]
            x  = solution.x[itt]
            y  = solution.y[itt]
            z  = solution.z[itt]
            vx = solution.vx[itt]
            vy = solution.vy[itt]
            vz = solution.vz[itt]

            #Get light reception angle and corresponding time delay
            deltat, light_vec = ode.deflection_position(ri=vec3([x,y,z]),rf=r_observer ,pncor=light_pncor, light_travel_time=light_travel_time)

            #Get corrected velocity (with redshift)
            v_redshift = ode.get_redshift_velocity(vec3([x,y,z]), r_observer, vec3([vx,vy,vz]), v_obs, sr_redshift=sr_redshift, gr_redshift=gr_redshift)
            
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
        time -= deltat_init 
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

    def get_sky_projection_fit_ts(self,
        #Orbital elements
        Omega, inc, omega, a, e, time_to_peri,
        #Scale parameters and distance
        m  = units.Rg, 
        R0 = units.R0/units.parsec,
        v_observer = [0., 0., 0.],
        #GR metric
        metric = schwarzschild_metric,
        r_transform = None,
        #Post-newtonian corrections
        orbit_pncor=True,
        light_pncor=True,
        light_travel_time=True,
        gr_redshift=True,
        sr_redshift=True,
        #Integration values
        tdata = None,                            
        interpolation_window = 0.4,             
        interpolation_window_npoints = 12,      
        ):
        """_summary_

        Args:
            Omega (float): Longitude of the ascending node in radians
            inc (float): Inclination in radians
            omega (float): argument of pericenter passage in radians
            a (float): semimajor axis in Astronomical Units
            e (float): eccentricity (must be between 0 and 1)
            time_to_peri (float): time to pericenter passage for the corresponding oscularing orbit
            m (float, optional): Central body's gravitational radius in meters. Defaults to that of SgrA*.
            R0 (float, optional): Distance between observer and central body. Defaults to GC distance in parsecs.
            v_observer (list, optional): Observer's velocity components. Defaults to [0., 0., 0.].            
            metric (norbit.metric, optional): Spacetime metric. Defaults to schwarzschild_metric.
            r_transform (lambda function, optional): Radial coordinate transformation (Deactivated) Defaults to None.
            orbit_pncor (bool, optional): If True, considers the first order Post-newtonian effects in the EOM. Defaults to True.
            light_pncor (bool, optional): If True, considers the first order Post-newtonian effects in the trajectory of photons connecting the orbiting body and the observer. Defaults to False.
            light_travel_time (bool, optional): If False, assumes speed of light to be infinite. Defaults to True.
            gr_redshift (bool, optional): If False, discards the gravitational field effects on the emmited photon's redshift. Defaults to True.
            sr_redshift (bool, optional): If False, discards the special relativity effects from the emitter's motion on the photon's redshift. Defaults to False.
            tdata (list): List of observational dates around which to interpolate the data.
            interpolation_window (float, optional): Time window around each date in tdata in days. Defaults to 0.4 days
            interpolation_window_npoints (int, optional): Number of interpolating points inside the time window. Defaults to 12.

        Returns:
            _output_fit: class with interpolated quantities for fitting
        """

        if tdata==None:
            raise ValueError("tdata must be a non-empty list")

        # Observer tetrad and position
        # Note: to match observational conventions, the observer is along negative part of the z-axis
        nr,nb,na = get_observer_tetrad( theta_g_deg = 0 , phi_g_deg=180.0)
    
        #Transform quantitites to dimensionless quantities
        a   *= units.astronomical_unit/m 
        R0  *= units.parsec/m

        time_to_peri *= units.year/(m/units.c)
        
        r_observer = -R0*nr 
        v_obs = vec3(v_observer)*1000/units.c
        rini, vini = get_position_and_velocity_at_t0(-time_to_peri,a,e,Omega,inc,omega)
                
        #Define integral problem
        ode = nPNsolver(
            initial_position= rini.values,
            initial_velocity= vini.values,
            metric=metric)

        #Time resolution for evaluations
        twindow   = interpolation_window*units.day/(m/units.c) 
        teval     = tdata*units.year/(m/units.c)
        
        sol = ode.integrate_fit(
            teval=teval,
            twindow=twindow, 
            npoints=interpolation_window_npoints,
            pncor=orbit_pncor)

        #Calculate romer delay for the initial point
        deltat_init, _ = ode.deflection_position(ri=rini,rf=r_observer ,pncor=light_pncor, light_travel_time=light_travel_time)

        #Retrieve the data
        time        = []
        alpha,beta  = [], []
        vrs         = []

        for itt in np.arange(0,len(sol.t)):

            #Get position and velocity
            t  = sol.t[itt]
            x  = sol.x[itt]
            y  = sol.y[itt]
            z  = sol.z[itt]
            vx = sol.vx[itt]
            vy = sol.vy[itt]
            vz = sol.vz[itt]

          
            #Get light reception angle and corresponding time delay
            deltat, light_vec = ode.deflection_position(ri=vec3([x,y,z]),rf=r_observer ,pncor=light_pncor, light_travel_time=light_travel_time)

            #Get corrected velocity (with redshift)
            v_redshift = ode.get_redshift_velocity(vec3([x,y,z]), r_observer, vec3([vx,vy,vz]), v_obs, sr_redshift=sr_redshift, gr_redshift=gr_redshift)
            
            #Append to lists
            time    .append(t+deltat)    
            alpha   .append( dot(light_vec, na)/dot(light_vec, nr) * units.rad_to_as)
            beta    .append( dot(light_vec, nb)/dot(light_vec, nr) * units.rad_to_as)
            vrs     .append(v_redshift)

            # This next line corresponds to the simplistic version of the projection when
            # the observer is exactly at infinity. It is only here for comparison purpouses
            # with the remaining codes.
            
            #alpha   .append( x/(r_observer.z)*units.rad_to_as)
            #beta    .append( -y/(r_observer.z)*units.rad_to_as)
            #vrs     .append(vz)


        #Convert time to years and distances to Astronomical units
        time =  np.array(time)
        time -= deltat_init 
        time *= (m/units.c)/units.year
        vrs = np.array(vrs)*units.c/1000.0

        #Return a class with functions
        return _output_fit(time,alpha,beta,vrs)


################################
# Old functions
################################

'''
def get_sky_projection(
        #Orbital elements
        Omega, inc, omega, a, e, 
        #Scale parameters and distance
        m  = units.Rg, 
        R0 = units.R0/units.parsec,
        #GR metric
        metric = schwarzschild_metric,
        #Post-newtonian corrections
        orbit_pncor=True,
        light_pncor=True,
        light_travel_time=True,
        gr_redshift=True,
        sr_redshift=False,
        #Integration values
        tol  = 4.5e-9,
        tmax = None,
        r_transform = None,
        time_resolution = 1.0,
        v_observer = [0., 0., 0.]
):
    """Returns the interpolating function for a given orbit as a function of the observer's time in arcseconds

    Args:
        Omega   (float): Longitude of the ascending node in radians
        inc     (float): Inclination in radians
        omega   (float): argument of pericenter passage in radians
        a       (float): semimajor axis in Astronomical Units
        e       (float): eccentricity (must be between 0 and 1)
        m       (float, optional): Central body's gravitational radius in meters. Defaults to that of SgrA*.
        R0      (float, optional): Distance between observer and central body. Defaults to GC distance in parsecs.
        metric  (norbit.metric, optional): Spacetime metric. Defaults to schwarzschild_metric.
        orbit_pncor (bool, optional): If True, considers the first order Post-newtonian effects in the EOM. Defaults to True.
        light_pncor (bool, optional): If True, considers the first order Post-newtonian effects in the trajectory of photons connecting the orbiting body and the observer. Defaults to False.
        light_travel_time (bool, optional): If False, assumes speed of light to be infinite. Defaults to True.
        gr_redshift (bool, optional): If False, discards the gravitational field effects on the emmited photon's redshift. Defaults to True.
        sr_redshift (bool, optional): If False, discards the special relativity effects from the emitter's motion on the photon's redshift. Defaults to False.
        tol (float, optional): Tolerance for the integrator. Defaults to 1e-10.
        tmax (float, optional): Maximum integration time in years. If None, uses the period of one single orbital revolution.
        r_transform (lambda function, optional): Radial coordinate transformation as a lambda function.
        time_resolution (float, optional): Time resolution of the integration in days. Defaults to 1.0.
        v_observer (list, optional): Observer's velocity with respect to the central body. Defaults to [0., 0., 0.].

    Returns:
        output: class with interpolated function
    """    
  
    # Observer tetrad and position
    # Note: to match observational conventions, the observer is along negative part of the z-axis
    nr,nb,na = get_observer_tetrad( theta_g_deg = 0 , phi_g_deg=180.0)
  
    #Transform quantitites to dimensionless quantities
    a   *= units.astronomical_unit/m 
    R0  *= units.parsec/m

    #Get the position and velocity vector for the apocenter position
    nr_apo, nv_apo          = get_apocenter_unit_vectors(Omega,inc,omega)
    r_apo_norm, v_apo_norm  = get_apocenter_position_and_velocity(a,e)
    
    #Radial coordinate transformation
    if r_transform != None:
        r_observer = -(r_transform(R0))*nr 
        r_apo_vec = r_transform(r_apo_norm)*nr_apo
        v_apo_vec = v_apo_norm*nv_apo
    else:
        r_observer = -R0*nr 
        r_apo_vec = r_apo_norm*nr_apo
        v_apo_vec = v_apo_norm*nv_apo

    #Maximum integration time in code units
    if tmax == None:
        tmax = kepler_period(a)
    else:
        tmax = tmax*units.year/(m/units.c)

    #Define integral problem
    ode = nPNsolver(
        initial_position= r_apo_vec.values,
        initial_velocity= v_apo_vec.values,
        metric=metric)

    #Time resolution for evaluations
    dt_eval = time_resolution*units.day/(m/units.c)
    solution = ode.integrate(tf=tmax, dt_eval=dt_eval, pncor=orbit_pncor)

    #Retrieve the data
    time        = []
    alpha,beta  = [], []
    xx,yy,zz    = [],[],[]
    vxx,vyy,vzz = [],[],[]
    vrs         = []

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
        deltat, light_vec = ode.deflection_position(ri=vec3([x,y,z]),rf=r_observer ,pncor=light_pncor, light_travel_time=light_travel_time)

        #Get corrected velocity (with redshift)
        v_obs = vec3(v_observer)*1000/units.c
        v_redshift = ode.get_redshift_velocity(vec3([x,y,z]), r_observer, vec3([vx,vy,vz]), v_obs, sr_redshift=sr_redshift, gr_redshift=gr_redshift)
        
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


   def get_sky_projection_fit(self,
        #Orbital elements
        Omega, inc, omega, a, e, 
        #Scale parameters and distance
        m  = units.Rg, 
        R0 = units.R0/units.parsec,
        v_observer = [0., 0., 0.],
        #GR metric
        metric = schwarzschild_metric,
        r_transform = None,
        #Post-newtonian corrections
        orbit_pncor=True,
        light_pncor=True,
        light_travel_time=True,
        gr_redshift=True,
        sr_redshift=True,
        #Integration values
        tmax = None,
        interpolation_window = 0.4,             #in days
        interpolation_window_npoints = 12,        
        tdata = None                            #in years
        ):

        # Observer tetrad and position
        # Note: to match observational conventions, the observer is along negative part of the z-axis
        nr,nb,na = get_observer_tetrad( theta_g_deg = 0 , phi_g_deg=180.0)
    
        #Transform quantitites to dimensionless quantities
        a   *= units.astronomical_unit/m 
        R0  *= units.parsec/m

        #Get the position and velocity vector for the apocenter position
        nr_apo, nv_apo          = get_apocenter_unit_vectors(Omega,inc,omega)
        r_apo_norm, v_apo_norm  = get_apocenter_position_and_velocity(a,e)
        
        #Radial coordinate transformation
        if r_transform != None:
            r_observer = -(r_transform(R0))*nr 
            r_apo_vec = r_transform(r_apo_norm)*nr_apo
            v_apo_vec = v_apo_norm*nv_apo
        else:
            r_observer = -R0*nr 
            r_apo_vec = r_apo_norm*nr_apo
            v_apo_vec = v_apo_norm*nv_apo

        #Maximum integration time in code units
        if tmax == None:
            tmax = kepler_period(a)
        else:
            tmax = tmax*units.year/(m/units.c)

        #Define integral problem
        ode = nPNsolver(
            initial_position= r_apo_vec.values,
            initial_velocity= v_apo_vec.values,
            metric=metric)

        #Time resolution for evaluations
        twindow   = interpolation_window*units.day/(m/units.c) 
        teval     = tdata*units.year/(m/units.c)

        solution = ode.integrate_fit(
            ti= 0.0, 
            tf=tmax+twindow, 
            twindow=twindow, 
            pncor=orbit_pncor, 
            teval=teval,
            npoints=interpolation_window_npoints)

        #Retrieve the data
        time        = []
        alpha,beta  = [], []
        vrs         = []

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
            deltat, light_vec = ode.deflection_position(ri=vec3([x,y,z]),rf=r_observer ,pncor=light_pncor, light_travel_time=light_travel_time)

            #Get corrected velocity (with redshift)
            v_obs = vec3(v_observer)*1000/units.c
            v_redshift = ode.get_redshift_velocity(vec3([x,y,z]), r_observer, vec3([vx,vy,vz]), v_obs, sr_redshift=sr_redshift, gr_redshift=gr_redshift)
            
            #Append to lists
            time    .append(t+deltat)    
            alpha   .append( dot(light_vec, na)/dot(light_vec, nr) * units.rad_to_as)
            beta    .append( dot(light_vec, nb)/dot(light_vec, nr) * units.rad_to_as)
            vrs     .append(v_redshift)

        #Convert time to years and distances to Astronomical units
        time =  np.array(time)
        time -= time[0] 
        time *= (m/units.c)/units.year

        vrs = np.array(vrs)*units.c/1000.0

        #Return a class with functions
        return _output_fit(time,alpha,beta,vrs)


'''