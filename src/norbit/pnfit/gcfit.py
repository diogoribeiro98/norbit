import numpy as np
from lmfit import Parameters
from lmfit import Minimizer

from ..physical_units import units
from ..pnutils.kepler_period import kepler_period
from ..pnutils.orbital_projection import orbital_projection
from .file_reading_utils import get_line_index, readlines_from_to

from ..pnutils.metric import minkowsky_metric, schwarzschild_metric

def p_weight(s):
    if s==0:
        return lambda r : r
    else:
        return lambda r: r*s/np.sqrt(r**2+s**2)

class nPNFitterGC:

    def __init__(self, fname):        

        #Load data from fname and see time range
        self.load_data_from_stefan_file(fname)

        #Get time range
        self.tmin, self.tmax, self.tdata_span = self.get_time_span()
        
        #Define the problem dimensions
        self.Rscale   = 1e6*(units.solar_mass*units.G/units.c**2)

        #Fitting auxiliart tools
        self.params =None
        self.orb = orbital_projection()
        
        #Create parameters according to chosen model
        self.params = Parameters()
        self.minimize_result = None
        self.minimize_sol = None

    #
    # Loading data functions
    #

    def load_data_from_stefan_file(self, fname):
        """
        Given a galactic center datafile with Stefan like formating, loads the associated data.
        All points must have at least the line

        ";	position	data	date	RA	delta	RA	DEC	delta	DEC"

        under which we should have all the "NACO like" data. Then the line

        ";	velocity	data	date	v	delta	v"

        under which we should find all the "SINFONI like" data. And optionally the line

        ";	GRAVITY	data"

        under which all gravity data should be. At the very end of the file we should find the line

        ";	full	error	matrix" 
        
        under which we should find the error matrix.

        All files contain the first two lines even if no data exists for those files. 

        Args:
            fname (string): name of file to load data from

        Returns: None
        """

        print('Loading data from: ', fname)

        #Storage arrays
        NACO    = []
        SINFONI = []
        GRAVITY = []

        #boolean variables
        has_naco   = True
        has_gravity= True
        has_sinfoni= True
        
        #Find index where those lines start
        index_list = np.array([ 
        get_line_index(';	position	data	date	RA	delta	RA	DEC	delta	DEC\n',fname),
        get_line_index(';	velocity	data	date	v	delta	v\n', fname),
        get_line_index(';	GRAVITY	data\n',fname),
        get_line_index(';	full	error	matrix\n',fname),
        ])
        
        #If no position data are found 
        if index_list[0] == None:
            raise ValueError("ERROR: Could not find position data in file. Make sure it is properly formatted")
        
        #If no velocity data are found
        if index_list[1] == None:
            raise ValueError("ERROR: Could not find velocity data in file. Make sure it is properly formatted")

        #Is there any GRAVITY data?
        if index_list[2] == None:
            print("No GRAVITY data. Loading NACO and SINFONI data")

            NACO = readlines_from_to(fname,index_list[0]+1,index_list[1])

            if index_list[3] == None:
                SINFONI = readlines_from_to(fname,index_list[1]+1,-1)
            else:
                SINFONI = readlines_from_to(fname,index_list[1]+1,index_list[3])
        
        else:
            NACO    = readlines_from_to(fname,index_list[0]+1,index_list[1])
            SINFONI = readlines_from_to(fname,index_list[1]+1,index_list[2])
        
            if index_list[3] == None:
                GRAVITY = readlines_from_to(fname,index_list[2]+1,-1)
            else:
                GRAVITY = readlines_from_to(fname,index_list[2]+1,index_list[3])
                
        NACO, SINFONI, GRAVITY = np.transpose(NACO), np.transpose(SINFONI), np.transpose(GRAVITY)

        #If no data in one of the arrays, make then None
        if len(NACO)==0:
            has_naco = False
            NACO=[[None], [None], [None], [None], [None]]
        if len(SINFONI)==0:
            has_sinfoni=False
            SINFONI=[[None],[None],[None]]
        if len(GRAVITY)==0:
            has_gravity=False
            GRAVITY=[[None], [None], [None], [None], [None]]

        #Store data
        self.astrometric_data = {
            "GRAVITY": {
                "hasData"   : has_gravity,
                "tdata"     : GRAVITY[0],
                "xdata"     : GRAVITY[1],
                "xdata_err" : GRAVITY[2],
                "ydata"     : GRAVITY[3],
                "ydata_err" : GRAVITY[4],
            },
            "NACO"   : {
                "hasData"   : has_naco,
                "tdata"     : NACO[0],
                "xdata"     : NACO[1],
                "xdata_err" : NACO[2],
                "ydata"     : NACO[3],
                "ydata_err" : NACO[4]
            },
        }

        self.spectroscopic_data = {
            "SINFONI": {
                "hasData"   : has_sinfoni,
                "tdata"     : SINFONI[0],
                "vdata"     : SINFONI[1],
                "vdata_err" : SINFONI[2]
            }
        }

        #Get elapsed time since first measurement in order
        self.teval = np.concatenate((self.astrometric_data['NACO']['tdata'],self.astrometric_data['GRAVITY']['tdata'],self.spectroscopic_data['SINFONI']['tdata']))
        self.teval = np.sort(np.delete(self.teval,np.where(self.teval==None)))
        
        return 

    #
    # Time span function
    #

    def get_time_span(self):

        tmin_list = []
        tmax_list = []

        for key in self.astrometric_data:

            if self.astrometric_data[key]['hasData']==False: 
                continue

            tmin_list.append(min(self.astrometric_data[key]['tdata'], default=float('inf')))
            tmax_list.append(max(self.astrometric_data[key]['tdata'], default=float('-inf')))
        
        for key in self.spectroscopic_data:
            
            if self.spectroscopic_data[key]['hasData']==False:
                continue
            
            tmin_list.append(min(self.spectroscopic_data[key]['tdata'], default=float('inf')))
            tmax_list.append(max(self.spectroscopic_data[key]['tdata'], default=float('-inf')))
        
        tmin = min(tmin_list)
        tmax = max(tmax_list)
        
        return tmin, tmax, (tmax - tmin)
    
    #
    # Fitting functions
    #

    def add_orbital_fit_parameters(self, Omega, inc, omega, sma, ecc, t0, x0, y0, vx, vy, vz, m ,r0):

        #Orbital parameters
        self.params.add('Omega'  , value = Omega[0]   , vary=Omega[3],  min=Omega[1], max = Omega[2])
        self.params.add('inc'    , value =   inc[0]   , vary=  inc[3],  min=  inc[1], max =   inc[2])
        self.params.add('omega'  , value = omega[0]   , vary=omega[3],  min=omega[1], max = omega[2])
        self.params.add('sma'    , value =   sma[0]   , vary=  sma[3],  min=  sma[1], max =   sma[2])
        self.params.add('ecc'    , value =   ecc[0]   , vary=  ecc[3],  min=  ecc[1], max =   ecc[2])
        
        #Drift and offset parameters
        self.params.add('x0'     , value =   x0[0], vary=    x0[3],  min=    x0[1], max =     x0[2])
        self.params.add('y0'     , value =   y0[0], vary=    y0[3],  min=    y0[1], max =     y0[2])
        self.params.add('vx'    , value =    vx[0], vary=    vx[3],  min=    vx[1], max =     vx[2])
        self.params.add('vy'    , value =    vy[0], vary=    vy[3],  min=    vy[1], max =     vy[2])
        self.params.add('vz'    , value =    vz[0], vary=    vz[3],  min=    vz[1], max =     vz[2])
        
        #GC 
        self.params.add('m'      , value =     m[0]   , vary=    m[3],  min=    m[1], max =     m[2])
        self.params.add('R0'     , value =    r0[0]   , vary=   r0[3],  min=   r0[1], max =    r0[2])

        # Adding time constraints is not as trivial as this
        # Check https://lmfit.github.io/lmfit-py/constraints.html
        # to use asteval functions and check
        # https://stackoverflow.com/questions/49931455/python-lmfit-constraints-a-b-c
        # to implement multi bound constraints
        #
        # From the way our modeling works, we need to have the following constrains
        #
        #  t0_data - Pkepler < t0 < t0_data
        #
        # meaning that the time of apocenter passage cannot happen after the first data point
        # and be at least one orbit before the first data point. This is convention.
        # The difference from the link above is that the left hand side term is dependend on
        # other parameters. Namely, the Kepler period depends on the semi-major axis and the
        # gravitational scale of the problem. We thus implement the following inequality:
        #
        # ti < t0 < tf <-> t0_data-Pkepler < t0 < t0_data
        #
        # and fit for the following two parameters (t0-ti) and (tf-t0). Since we know that these
        # two parameters cannot be smaller than 0, we can contrain them accordingly. The individual
        # parameters are then obtained from the associated expressions.
        # 
    
        #Initial guesses
        Pkepler = kepler_period( sma[0]*units.astronomical_unit/(m[0]*self.Rscale) )*(m[0]*self.Rscale)/units.c/units.year
        t0_minus_ti_ival = t0[0] - (self.tmin - Pkepler)
        tf_minus_t0_ival = self.tmin - t0[0] 
        
        #Helper quantities for fit
        def kperiod(sma,m):
            return kepler_period( sma*units.astronomical_unit/(m*self.Rscale) )*(m*self.Rscale)/units.c/units.year

        self.params._asteval.symtable['tf'] = self.tmin
        self.params._asteval.symtable['kperiod'] = kperiod
    
        self.params.add('t0_minus_ti', value=t0_minus_ti_ival,  vary=t0[3], min=0)
        self.params.add('tf_minus_t0', value=tf_minus_t0_ival,  vary=t0[3], min=0)

        self.params.add('ti', expr='tf - kperiod(sma,m)')
        self.params.add('t0', expr='ti + t0_minus_ti',max=self.tmin)

        return
    
    def add_orbital_fit_parameters_tperi(self, Omega, inc, omega, sma, ecc, tperi, x0, y0, vx, vy, vz, m ,r0, tosc):

        #Orbital parameters
        self.params.add('Omega'  , value = Omega[0]   , vary=Omega[3],  min=Omega[1], max = Omega[2])
        self.params.add('inc'    , value =   inc[0]   , vary=  inc[3],  min=  inc[1], max =   inc[2])
        self.params.add('omega'  , value = omega[0]   , vary=omega[3],  min=omega[1], max = omega[2])
        self.params.add('sma'    , value =   sma[0]   , vary=  sma[3],  min=  sma[1], max =   sma[2])
        self.params.add('ecc'    , value =   ecc[0]   , vary=  ecc[3],  min=  ecc[1], max =   ecc[2])
        
        #Drift and offset parameters
        self.params.add('x0'     , value =   x0[0], vary=    x0[3],  min=    x0[1], max =     x0[2])
        self.params.add('y0'     , value =   y0[0], vary=    y0[3],  min=    y0[1], max =     y0[2])
        self.params.add('vx'    , value =    vx[0], vary=    vx[3],  min=    vx[1], max =     vx[2])
        self.params.add('vy'    , value =    vy[0], vary=    vy[3],  min=    vy[1], max =     vy[2])
        self.params.add('vz'    , value =    vz[0], vary=    vz[3],  min=    vz[1], max =     vz[2])
        
        #GC 
        self.params.add('m'      , value =     m[0]   , vary=    m[3],  min=    m[1], max =     m[2])
        self.params.add('R0'     , value =    r0[0]   , vary=   r0[3],  min=   r0[1], max =    r0[2])

        # Adding time constraints is not as trivial as this
        # Check https://lmfit.github.io/lmfit-py/constraints.html
        # to use asteval functions and check
        # https://stackoverflow.com/questions/49931455/python-lmfit-constraints-a-b-c
        # to implement multi bound constraints
        #
        # From the way our modeling works, we need to have the following constrains
        #
        #  tosc < tperi < tosc + Pkepler
        #
        # The Kepler period depends on the semi-major axis and the
        # gravitational scale of the problem. We thus implement the following inequality:
        #
        # ti < t0 < tf <-> tosc < tperi < tosc + Pkepler
        #
        # and fit for the following two parameters (t0-ti) and (tf-t0). Since we know that these
        # two parameters cannot be smaller than 0, we can contrain them accordingly. The individual
        # parameters are then obtained from the associated expressions.
        # 
    
        #Add tosculating
        self.params.add('tosc', value=tosc, vary=False)

        #Initial guesses
        Pkepler = kepler_period( sma[0]*units.astronomical_unit/(m[0]*self.Rscale) )*(m[0]*self.Rscale)/units.c/units.year
        t0_minus_ti_ival = tperi[0]- tosc
        tf_minus_t0_ival = tosc + Pkepler - tperi[0] 
        
        #Helper quantities for fit
        def kperiod(sma,m):
            return kepler_period( sma*units.astronomical_unit/(m*self.Rscale) )*(m*self.Rscale)/units.c/units.year

        self.params._asteval.symtable['tf'] = self.tmin
        self.params._asteval.symtable['kperiod'] = kperiod
    
        self.params.add('t0_minus_ti', value=t0_minus_ti_ival,  vary=tperi[3], min=0)
        self.params.add('tf_minus_t0', value=tf_minus_t0_ival,  vary=tperi[3], min=0)

        self.params.add('tf', expr='tosc + kperiod(sma,m)')
        self.params.add('tperi', expr='tf - tf_minus_t0')

        return

    def residuals(self, params, model, 
                  priors = {}, 
                  ignore_NACO=False, 
                  ignore_GRAVITY=False, 
                  ignore_SINFONI=False, 
                  s_weight=0.0,
                  fit_window=1.0,
                  window_npoints=10):

        #Get orbital parameters
        orbital_params = {
        "Omega" : params['Omega']*np.pi/180,
        "inc"   : params['inc']  *np.pi/180,
        "omega" : params['omega']*np.pi/180,
        "a"     : params['sma']  ,
        "e"     : params['ecc']  ,
        }

        #Offset parameters
        x0      = params['x0']
        y0      = params['y0']
        vx      = params['vx']
        vy      = params['vy']
        vz      = params['vz']

        #GC quantities
        m = params['m']*self.Rscale

        gc_params = {
        "m"  : m,
        "R0" : params['R0'],
        'v_observer' : [(-3.156e-3*units.as_to_rad/units.year)*params['R0']*units.parsec/units.kilometer,(-5.585e-3*units.as_to_rad/units.year)*params['R0']*units.parsec/units.kilometer,vz]
        #'v_observer' : [0.0, 0.0, 0.0]
        }
            
        #Find integration times
        tosc = params['tosc']
        time_to_peri  = (params['tperi']-tosc)
        teval = (self.teval - tosc)
        
        #Get model data depending on the physical setup
        match model:
            
            case 'Newton':
                #Purely Newtonian orbit without light travel time
                sol = self.orb.get_sky_projection_fit_ts(   
                    **orbital_params, **gc_params,
                    time_to_peri=time_to_peri,
                    orbit_pncor=False,
                    light_pncor=False,
                    light_travel_time=False,
                    gr_redshift=False,
                    sr_redshift=False,
                    metric=minkowsky_metric,
                    interpolation_window = fit_window,             #in days
                    interpolation_window_npoints= window_npoints, #in days
                    tdata = teval)
            
            case 'Newton+Romer':
                # Newtonian orbit with light travel time
                sol = self.orb.get_sky_projection_fit_ts(   
                    **orbital_params, **gc_params,
                    time_to_peri=time_to_peri,
                    orbit_pncor=False,
                    light_pncor=False,
                    light_travel_time=True,
                    gr_redshift=False,
                    sr_redshift=False,
                    metric=minkowsky_metric,
                    interpolation_window = fit_window,             #in days
                    interpolation_window_npoints= window_npoints, #in days
                    tdata = teval)
            
            case 'Newton+Romer+SR':
                # Newtonian orbit with light travel time and special relativity effects
                sol = self.orb.get_sky_projection_fit_ts(   
                    **orbital_params, **gc_params,
                    time_to_peri=time_to_peri,
                    orbit_pncor=False,
                    light_pncor=False,
                    light_travel_time=True,
                    gr_redshift=False,
                    sr_redshift=True,
                    metric=minkowsky_metric,
                    interpolation_window = fit_window,             #in days
                    interpolation_window_npoints= window_npoints, #in days
                    tdata = teval)
            
            case 'Schwarzschild':
                #Full 1PN equations of motion for Schwarzschild coordinates
                sol = self.orb.get_sky_projection_fit_ts(   
                    **orbital_params, **gc_params,
                    time_to_peri=time_to_peri,
                    orbit_pncor=True,
                    light_pncor=True,
                    light_travel_time=True,
                    gr_redshift=True,
                    sr_redshift=True,
                    metric=schwarzschild_metric,
                    interpolation_window = fit_window,             #in days
                    interpolation_window_npoints= window_npoints, #in days
                    tdata = teval)
            
            case 'coco':
                #Code comparison mode
                sol = self.orb.get_sky_projection_fit_code_comparison(
                    **orbital_params, **gc_params,
                    time_to_peri=time_to_peri,
                    orbit_pncor=False,
                    light_pncor=False,
                    light_travel_time=True,
                    gr_redshift=True,
                    sr_redshift=True,
                    metric=minkowsky_metric,
                    interpolation_window = fit_window,             
                    interpolation_window_npoints= window_npoints, 
                    tdata = teval )


            case _:
                raise ValueError('CRITICAL ERROR: How did you get here?')
            
        self.minimize_sol = sol
        
        fx = sol.RA
        fy = sol.DEC
        fv = sol.vrs

        #
        # Calculate residuals vector fromd data
        #

        p = p_weight(s_weight)
        
        residuals_vector_x = 0
        residuals_vector_y = 0
        residuals_vector_v = 0

        if self.astrometric_data['GRAVITY']['hasData'] == True and ignore_GRAVITY==False:
        
            xmodel_gravity = -fx((self.astrometric_data['GRAVITY']['tdata'] - tosc))  
            ymodel_gravity =  fy((self.astrometric_data['GRAVITY']['tdata'] - tosc))  
        
            residuals_vector_x_gravity =  p((xmodel_gravity-self.astrometric_data['GRAVITY']['xdata'])/self.astrometric_data['GRAVITY']['xdata_err'])
            residuals_vector_y_gravity =  p((ymodel_gravity-self.astrometric_data['GRAVITY']['ydata'])/self.astrometric_data['GRAVITY']['ydata_err'])
        else:
            residuals_vector_x_gravity = []
            residuals_vector_y_gravity = []

        if self.astrometric_data['NACO']['hasData'] == True and ignore_NACO==False:

            xmodel_naco = -fx((self.astrometric_data['NACO']['tdata'] - tosc))  
            ymodel_naco =  fy((self.astrometric_data['NACO']['tdata'] - tosc))  
            
            #Drift in data
            xdata_drift = self.astrometric_data['NACO']['xdata'] - ( x0 + vx * (self.astrometric_data['NACO']['tdata']-2009.02))
            ydata_drift = self.astrometric_data['NACO']['ydata'] - ( y0 + vy * (self.astrometric_data['NACO']['tdata']-2009.02))
            residuals_vector_x_naco =  p((xmodel_naco-xdata_drift)/self.astrometric_data['NACO']['xdata_err'])
            residuals_vector_y_naco =  p((ymodel_naco-ydata_drift)/self.astrometric_data['NACO']['ydata_err'])
            
        else:
            residuals_vector_x_naco = []
            residuals_vector_y_naco = []

        if self.spectroscopic_data['SINFONI']['hasData'] == True and ignore_SINFONI==False:
            vmodel_sinfoni = fv((self.spectroscopic_data['SINFONI']['tdata'] - tosc)) 
            residuals_vector_v_sinfoni = p((vmodel_sinfoni-(self.spectroscopic_data['SINFONI']['vdata']))/self.spectroscopic_data['SINFONI']['vdata_err'])
        
        else:
            residuals_vector_v_sinfoni = []

        residuals_vector_x = np.concatenate((residuals_vector_x_gravity, residuals_vector_x_naco),axis=0)
        residuals_vector_y = np.concatenate((residuals_vector_y_gravity, residuals_vector_y_naco),axis=0)
        residuals_vector_v = residuals_vector_v_sinfoni

        #
        # Residuals from priors
        #

        prior_chi2 = []
        for prior in priors.values():
            for idx, parameter in enumerate(prior['parameter']):
                prior_chi2.append( (params[parameter] - prior['prior'][idx])/prior['error'][idx] )

        
        residuals_vector = np.concatenate((residuals_vector_x,residuals_vector_y,residuals_vector_v, prior_chi2 ),axis=0)
        
        return residuals_vector

    def find_minimum(self, 
                    model='Newton', 
                    niter=50, 
                    ignore_NACO=False, 
                    ignore_GRAVITY=False, 
                    ignore_SINFONI=False, 
                    s_weight=0.0, 
                    priors = {},
                    method='leastsq',
                    window=2.0,                     
                    window_npoints=10):

        #print('Starting finding minimum routine')
        implemented_models = ('coco', 
                              'Newton', 
                              'Newton+Romer', 
                              'Newton+Romer+SR', 
                              'Schwarzschild')
        
        if model in implemented_models:

            func_args = {   'model': model, 
                            'ignore_NACO': ignore_NACO, 
                            'ignore_GRAVITY': ignore_GRAVITY, 
                            'ignore_SINFONI': ignore_SINFONI, 
                            's_weight': s_weight,
                            'priors' : priors, 
                            'window_npoints': window_npoints,
                            'fit_window': window
                        }

            fitter = Minimizer( self.residuals, 
                                self.params, 
                                max_nfev=niter, 
                                fcn_kws=func_args)
            
            #MCMC
            #self.minimize_result = fitter.minimize(method = 'emcee', burn=10, nwalkers=100, steps=20, thin=1, is_weighted=True, progress=True,
            #run_mcmc_kwargs={'skip_initial_state_check':True})
            
            #Chisquared
            self.minimize_result =fitter.minimize(method = method)
            
            return self.minimize_result 
        else:
            raise ValueError('ERROR: No model called {}. Allowed values are {}'.format(model,implemented_models))

    def rerun(  self, 
                model='Newton', 
                niter=50, 
                ignore_NACO=False, 
                ignore_GRAVITY=False, 
                ignore_SINFONI=False, 
                s_weight=0.0,
                method='leastsq',
                window=2.0,                     
                window_npoints=10):
        
        func_args = {  'model': model, 
                        'ignore_NACO': ignore_NACO, 
                        'ignore_GRAVITY': ignore_GRAVITY, 
                        'ignore_SINFONI': ignore_SINFONI, 
                        's_weight': s_weight, 
                        'window_resolution': window_npoints,
                        'fit_window': window
                        }

        fitter = Minimizer( self.residuals, 
                            self.minimize_result.params, 
                            max_nfev=niter, 
                            fcn_kws= func_args)
        self.minimize_result =fitter.minimize(method = method)

        return self.minimize_result 

    #====================
    # Not used functions
    #====================

    #
    # Loading data functions
    #

    def load_astrometric_data(self, fname, instrument):
        
        try:
            data = np.loadtxt(fname)
        except:
            return

        self.astrometric_data[instrument]['tdata']     = np.array(data[:,0])
        self.astrometric_data[instrument]['xdata']     = np.array(data[:,1])
        self.astrometric_data[instrument]['xdata_err'] = np.array(data[:,2])
        self.astrometric_data[instrument]['ydata']     = np.array(data[:,3])
        self.astrometric_data[instrument]['ydata_err'] = np.array(data[:,4])

        return 
    
    def load_spectroscopic_data(self,fname,instrument):
        
        try:
            data = np.loadtxt(fname)
        except:
            return

        self.spectroscopic_data[instrument]['tdata']     = np.array(data[:,0])
        self.spectroscopic_data[instrument]['vdata']     = np.array(data[:,1])
        self.spectroscopic_data[instrument]['vdata_err'] = np.array(data[:,2])

        return
