import numpy as np
from numba import njit
from CyRK import nbsolve_ivp
from CyRK import pysolve_ivp

#Note: nbsolve_ivp cannot do backwards integration so we need to use pysolve_ivp for that part of the integration

#########################################
# Light travel and deflection functions
#########################################

@njit
def deflection_position_numba(
        ri, 
        rf,
        pn_coefficients_1st_order = [-1,-2, 3, 2],
        pncor=True, 
        light_travel_time=True):
        """ Given an initial and final position, calculates the observed position on sky for the specific pn parameters.
        """

        #Distance vector from observer to emission point
        D = (rf-ri)
        Dnorm = np.linalg.norm(D)
        nD = D/Dnorm

        if pncor == False:
            return int(light_travel_time)*Dnorm , nD  
        else:

            #Get coefficients
            T1 = pn_coefficients_1st_order[0]
            V1 = pn_coefficients_1st_order[1]
            N1 = pn_coefficients_1st_order[2]
            H1 = pn_coefficients_1st_order[3]

            #Angular momentum of photon orbit
            L     = np.cross(ri,rf)
            #Lnorm = norm(L)

            #Impact parameter
            b = np.cross(D , L)/Dnorm**2
            bnorm = np.linalg.norm(b)

            #Normalized vectors or observer and emitter
            rinorm = np.linalg.norm(ri)
            rfnorm = np.linalg.norm(rf)
            nri    = ri/rinorm
            nrf    = rf/rfnorm

            #0th order term (Euclidean)
            pn0 = nD

            #1st order corrections
            pn1_dx1_parallel        =  ( -(T1 + V1 + N1 + H1)/rfnorm + (N1/3) * (bnorm**2/rfnorm**3) ) * nD
            pn1_dx1_perpendicular   =  (  (T1 + V1)*( np.dot(nrf,nD) + 1 ) + (N1/3)*(np.dot(nrf,nD)**3 +1) )*b/bnorm**2

            pn1_x1_perpendicular    = -(1/Dnorm)*(  (T1 + V1 +N1/3)*( rfnorm - rinorm + Dnorm )/bnorm**2 + (N1/3)*(1/rfnorm - 1/rinorm) )*b
            pn1_x1_parallel_scalar  = -( -(T1 + V1 + N1 + H1)*np.log( (rfnorm/rinorm)*( np.dot(nrf,nD) + 1 )/( np.dot(nri,nD) + 1 )) + (N1/3) * ( np.dot(nrf,nD) - np.dot(nri,nD))  )

            #Light deflection vector at the observer position
            dxdtau    = pn0 + pn1_dx1_parallel + pn1_dx1_perpendicular + pn1_x1_perpendicular
            
            #Time interval between emission and absorption
            delta_tau = Dnorm + pn1_x1_parallel_scalar

            return int(light_travel_time)*delta_tau , dxdtau 

@njit
def dAdr_numba(
          r,v,sigma,
          pn_coefficients_1st_order):    
        
        #Define the pn parameters
        T1 = pn_coefficients_1st_order[0]
        V1 = pn_coefficients_1st_order[1]
        N1 = pn_coefficients_1st_order[2]
        H1 = pn_coefficients_1st_order[3]

        n = r/np.linalg.norm(r)
        t1 = (N1/3)*np.dot(v, np.cross(n,np.cross(sigma,n)))/np.linalg.norm(r)
        t2 = -(T1 + V1 + N1 + H1)* np.dot(v,sigma+n)/( np.linalg.norm(r) *(1+np.dot(n,sigma)))

        return t1+t2

@njit  
def dtdt0_numba(ri,rf,vi,vf,
                pn_coefficients_1st_order):
        
        D = (rf-ri)
        Dnorm = np.linalg.norm(D)
        nD = D/Dnorm

        return 1/(1 - np.dot(nD,vf-vi)+ (dAdr_numba(rf,vf,nD,pn_coefficients_1st_order) - dAdr_numba(ri,vi,nD,pn_coefficients_1st_order)) )  

@njit   
def get_redshift_velocity_numba(
          ri,rf,vi,vf, 
          sr_redshift=True, 
          gr_redshift=True):

        D = (rf-ri)
        Dnorm = np.linalg.norm(D)
        nD = D/Dnorm

        #If no GR nor SR effects are considered, return the 'galilean' redshift
        if gr_redshift==False and sr_redshift==False:
            return np.dot(nD,vf-vi)

        #If we wish to consider uniquely the GR effects we need only evaluate the difference in the gtt metric components
        elif gr_redshift==True and sr_redshift==False:
            
            #Term associated with the initial position
            r  = np.linalg.norm(ri)
            v   = np.linalg.norm(vi)

            nr = ri/np.linalg.norm(ri) 

            A = 1-2/r
            B = 1
            D = (2/r)/(1-2/r)

            dsdt_i = -A+ v**2*B + np.dot(nr,vi)**2*D

            #Term associated with the final position
            r  = np.linalg.norm(rf)
            v   = np.linalg.norm(vf)

            nr = rf/np.linalg.norm(rf) 

            A = 1-2/r
            B = 1
            D = (2/r)/(1-2/r)

            dsdt_f = -A+ v**2*B + np.dot(nr,vf)**2*D

            return np.sqrt(dsdt_f/dsdt_i)-1

        #If only the SR effects are wanted, we use the relativistic dopler formula (note that it is very close to the galilean one for low velocities)
        elif gr_redshift==False and sr_redshift==True:
            
             #Term associated with the initial position
            r  = np.linalg.norm(ri)
            v   = np.linalg.norm(vi)

            nr = ri/np.linalg.norm(ri) 

            A = 1
            B = 1
            D = 1

            dsdt_i = -A+ v**2*B + np.dot(nr,vi)**2*D

            #Term associated with the final position
            r  = np.linalg.norm(rf)
            v   = np.linalg.norm(vf)

            nr = rf/np.linalg.norm(rf) 

            A = 1
            B = 1
            D = 1

            dsdt_f = -A+ v**2*B + np.dot(nr,vf)**2*D

            return np.sqrt(dsdt_f/dsdt_i)/(1 - np.dot(nD,vf-vi))-1

        #If all effects are considered, we must evaluate the full expression for the redshift
        else:
            
            #Term associated with the initial position
            r  = np.linalg.norm(ri)
            v   = np.linalg.norm(vi)

            nr = ri/np.linalg.norm(ri) 

            A = 1-2/r
            B = 1
            D = (2/r)/(1-2/r)

            dsdt_i = -A+ v**2*B + np.dot(nr,vi)**2*D

            #Term associated with the final position
            r  = np.linalg.norm(rf)
            v   = np.linalg.norm(vf)

            nr = rf/np.linalg.norm(rf) 

            A = 1-2/r
            B = 1
            D = (2/r)/(1-2/r)

            dsdt_f = -A+ v**2*B + np.dot(nr,vf)**2*D

            return np.sqrt(dsdt_f/dsdt_i)*dtdt0_numba(ri,rf,vi,vf,[-1,-2, 3, 2])-1

@njit
def EOM_newton(t, state):
    r = state[0:3]
    v = state[3:6]
    Fnewton = - r / np.linalg.norm(r)**3
    Force = Fnewton
    return np.hstack((v,Force))

@njit
def EOM_schwarzschild(t, state):
    r = state[0:3]
    v = state[3:6]
    
    rnorm = np.linalg.norm(r)
    vnorm = np.linalg.norm(v)
        
    _ ,V1,N1,H1 = [-1.,-2., 3., 2.]
    T2, _, _, _ = [ 2., 0., 2., 4.]
    
    Fnewton = - r / np.linalg.norm(r)**3
    ForcePN1 = 1/rnorm**3 *( (T2/rnorm + V1 *vnorm**2  + N1*(np.dot(r , v)/rnorm)**2 )*r + H1 * np.dot(r,v)*v)                     

    Force = Fnewton + ForcePN1
    return np.hstack((v,Force))

def integrate_fit_numba(  EOM,
                    rini,
                    vini, 
                    teval,
                    twindow = 80,
                    npoints = 24, 
                    rtol=1e-13, 
                    atol=1e-20):


        #Create list of evaluation points around input times
        window   = np.linspace(-twindow,twindow,2*npoints+1)
        
        tlist = []
        for t in teval:
            tpoints = t + window  
            tlist.extend(tpoints)

        #Sort values
        tsorted = np.asarray(sorted(tlist))

        #Check if backwards integration is needed
        if (min(tsorted) < 0.0) & (max(tsorted) > 0.0):
            #print('Requires backward and forward integration')

            neg_tspan = ( 0.0, min(tsorted) )
            pos_tspan = ( 0.0, max(tsorted) )
 
            #Split sorted times into negative and positive values
            neg_teval = tsorted[tsorted<0]
            pos_teval = tsorted[tsorted>=0]

            #Revert negative values and do backwards integration
            #print('Performin backward integration')

            neg_teval = -np.sort(-neg_teval)
            neg_result = pysolve_ivp(
                    EOM_newton, 
                    neg_tspan ,
                    np.asarray([ rini[0] , rini[1] , rini[2] , vini[0] , vini[1] , vini[2] ]),                    
                    method="RK45",
                    t_eval=neg_teval,
                    rtol=rtol,
                    atol=atol)

            #print("Was Integration was successful?", neg_result.success)
            #Integrate forward
            #print('Performin forward integration')
            pos_result = nbsolve_ivp(
                    EOM_newton, 
                    pos_tspan ,
                    [ rini[0] , rini[1] , rini[2] , vini[0] , vini[1] , vini[2] ],                    
                    rk_method = 1,
                    t_eval=pos_teval,
                    rtol=rtol,
                    atol=atol)    

            #Concatenate the points from both solutions
            t = np.hstack((neg_result.t,pos_result.t))
            y = np.hstack((neg_result.y,pos_result.y))
            
            y = y[:,t.argsort()]
            t = np.sort(t)
        
            return t, y

        #If not forward integration suffices
        elif (min(tsorted) < 0.0) & (max(tsorted) < 0.0):
            print('ONLY backward integration required')
            
            tspan = ( 0.0 , min(tsorted) )

            #Revert negative values and do backwards integration
            tsorted = -np.sort(-tsorted)

            result = nbsolve_ivp(
                    EOM, 
                    tspan ,
                    [ rini[0] , rini[1] , rini[2] , vini[0] , vini[1] , vini[2] ],                    
                    method='RK45',
                    t_eval=tsorted,
                    rtol=rtol,
                    atol=atol,
                    dense_output=False)

            #reorder negative solution
            y = result.y[:,result.t.argsort()]
            t = np.sort(result.t)

            return t,y

        elif (min(tsorted) > 0.0) & (max(tsorted) > 0.0):

            print('NO backward integration required')

            tspan = (     0.0     , max(tsorted) )

            result = nbsolve_ivp(
                    EOM, 
                    tspan ,
                    [ rini[0] , rini[1] , rini[2] , vini[0] , vini[1] , vini[2] ],                    
                    method='RK45',
                    t_eval=tsorted,
                    rtol=rtol,
                    atol=atol,
                    dense_output=False)
            
            return result.t, result.y

        else:
            raise ValueError('CRITICAL ERROR: How did you get here?')
        

######################################
# Sky position functions for fitting
######################################
from ..physical_units import units
from ..pnutils.metric import schwarzschild_metric
from ..pnutils.orbital_elements import get_position_and_velocity_at_t0
from ..pnutils.observer_tetrad import get_observer_tetrad
from ..pnutils.orbital_projection import _output_fit

#Helper output function 
@njit
def get_solution_output(
        solt,sol,
        r_observer,v_observer,
        pn_coefficients_1st_order,
        m,
        light_pncor,
        light_travel_time,
        sr_redshift,
        gr_redshift,
        na,nb,nr,
        dt_init):
    
    time        = np.empty_like(solt)
    alpha,beta  = np.empty_like(solt), np.empty_like(solt)
    vrs         = np.empty_like(solt)

    for itt in np.arange(0,len(solt)):

        #Get position and velocity
        t  = solt[itt]
        rini = sol[0:3,itt]
        vini = sol[3:6,itt]
        
        #Get light reception angle and corresponding time delay
        deltat, light_vec = deflection_position_numba(
            ri=np.asarray(rini),
            rf=np.asarray(r_observer) ,
            pn_coefficients_1st_order=pn_coefficients_1st_order,
            pncor=light_pncor, 
            light_travel_time=light_travel_time)
        
        #Get corrected velocity (with redshift)
        v_redshift = get_redshift_velocity_numba(rini, r_observer, vini, v_observer, sr_redshift=sr_redshift, gr_redshift=gr_redshift)
        
        #Append to lists
        time[itt] = t+deltat   
        alpha[itt] =  np.dot(light_vec, na)/np.dot(light_vec, nr) * units.rad_to_as
        beta[itt] =  np.dot(light_vec, nb)/np.dot(light_vec, nr) * units.rad_to_as
        vrs [itt] = v_redshift
        
    #Convert time to years and distances to Astronomical units
    time -= dt_init 
    time *= (m/units.c)/units.year
    vrs  *= units.c/1000.0

    return time, alpha, beta, vrs

#Sky projection (not compiled with numba)
def get_sky_projection_fit_ts_numba(
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
        """
        Returns the interpolating function for a given orbit as a function of the observer's time. 

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

        # Observer tetrad and position
        # Note: to match observational conventions, the observer is along negative part of the z-axis
        nr,nb,na = get_observer_tetrad( theta_g_deg = 0 , phi_g_deg=180.0)
    
        #Transform quantitites to dimensionless quantities
        a   *= units.astronomical_unit/m 
        R0  *= units.parsec/m

        time_to_peri *= units.year/(m/units.c)
        
        r_observer = -R0*nr 
        v_obs = v_observer*1000/units.c
        rini, vini = get_position_and_velocity_at_t0(-time_to_peri,a,e,Omega,inc,omega)

        #Integrate problem
        

        #Define integral problem
        #ode = nPNsolver(
        #    initial_position= rini,
        #    initial_velocity= vini,
        #    metric=metric)

        #Time resolution for evaluations
        twindow   = interpolation_window*units.day/(m/units.c) 
        teval     = tdata*units.year/(m/units.c)
        
        solt, sol= integrate_fit_numba(
            EOM_schwarzschild,
            rini,
            vini,
            teval=teval,
            twindow=twindow, 
            npoints=interpolation_window_npoints
        )

        #Calculate romer delay for the initial point
        deltat_init, _ = deflection_position_numba(
            ri=rini,
            rf=r_observer ,
            pn_coefficients_1st_order=metric.pn_coefficients_1st_order,
            pncor=light_pncor, 
            light_travel_time=light_travel_time)

    
        time,alpha,beta,vrs = get_solution_output(
            solt=solt,
            sol=sol,
            r_observer=r_observer,
            v_observer=v_obs,
            pn_coefficients_1st_order=metric.pn_coefficients_1st_order,
            m=m,
            light_pncor=light_pncor,
            light_travel_time=light_travel_time,
            sr_redshift=sr_redshift,
            gr_redshift=gr_redshift,
            na=na,nb=nb,nr=nr,
            dt_init=deltat_init)

        return _output_fit(time,alpha,beta,vrs)
