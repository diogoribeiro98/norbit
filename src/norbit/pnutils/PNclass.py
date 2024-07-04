import numpy as np 
from scipy.integrate import solve_ivp

from ..vector.vector_class import vec3
from ..vector.vector_functions import dot, norm, cross

class nPNsolver:

    #Constructor
    def __init__(self , 
                 initial_position,
                 initial_velocity,
                 pn_coefficients_1st_order = np.array([-1,-2, 3, 2]),
                 pn_coefficients_2nd_order = np.array([ 2, 0, 2, 4]),
                 pn_coefficients_3rd_order = np.array([ 0, 0, 4, 8]),
                 pncor = True,
                 tol = 1e-8):
        
        #===============
        # Setup routine
        #===============

        #Problem scale
        self.m = 1.0

        #Integration tolerance
        self.tol = tol

        #Define the pn parameters
        self.T1 = pn_coefficients_1st_order[0]
        self.V1 = pn_coefficients_1st_order[1]
        self.N1 = pn_coefficients_1st_order[2]
        self.H1 = pn_coefficients_1st_order[3]

        self.T2 =  pn_coefficients_2nd_order[0]
        self.V2 =  pn_coefficients_2nd_order[1]
        self.N2 =  pn_coefficients_2nd_order[2]
        self.H2 =  pn_coefficients_2nd_order[3]

        self.T3 =  pn_coefficients_3rd_order[0]
        self.V3 =  pn_coefficients_3rd_order[1]
        self.N3 =  pn_coefficients_3rd_order[2]
        self.H3 =  pn_coefficients_3rd_order[3]

        #Initial position
        self.r = vec3(initial_position)
        self.v = vec3(initial_velocity)

        #Define the associated force terms
        if pncor == False:
            self.Force = self.newtonForce
        else:
            self.Force = self.grForce

    #==========
    # Methods
    #==========

    def newtonForce(self):
        """ Classical Newtonian force law
        """
        Fnewton = - self.m * self.r / self.r.norm()**3
        return Fnewton

    def pnForce(self):
        """PN order corrections to classical newtonian force
        """
        rnorm = self.r.norm()
        vnorm = self.v.norm()
        ForcePN1 = 1/rnorm**3 *( (self.T2/rnorm + self.V1 *vnorm**2  + self.N1*(dot(self.r , self.v)/rnorm)**2 )*self.r + self.H1 * dot(self.r , self.v) *self.v)                     
        #ForcePN2 = 1/rnorm**4 *( (self.T3/rnorm + self.V2 *vnorm**2  + self.N2*(dot(self.r , self.v)/rnorm)**2 )*self.r + self.H2 * dot(self.r , self.v) *self.v)                     
        
        return ForcePN1 #+ForcePN2
    
    def grForce(self):
        """ Newtonian + 1PN correction force
        """
        return self.newtonForce() + self.pnForce()

    def EOM(self,t, state):
        """ Equation of motion for massive particles
        """
        
        x, y, z    = state[0] , state[1] , state[2] 
        vx, vy, vz = state[3] , state[4] , state[5]

        self.r = vec3([x,y,z])
        self.v = vec3([vx,vy,vz])
        
        Force = self.Force()

        return [ vx , vy , vz, Force.x, Force.y, Force.z ]
 
    def integrate(self, tf = 2e6  , dt_eval = 80.0 ):
        """
        Integrate Equation of motion given the initial conditions of the problem. 
        The default final time is 1 year in dimensionless units for a central mass of 4 billion solar masses (SrgA*). The sampling of points is 30 minutes in dimensionless units for the same mass.
        """

        t_span = (0.0, tf+dt_eval)
        t = np.arange(0.0, tf+dt_eval, dt_eval)

        result = solve_ivp(
                    self.EOM, 
                    t_span, 
                    [ self.r.x , self.r.y , self.r.z , self.v.x , self.v.y , self.v.z  ],
                    method= 'RK45' ,
                    t_eval=t,
                    rtol = self.tol)
        
        return result
    
    def deflection_position(self, ri , rf, pncor = True):
        """ Given an initial and final position, calculates the observed position on sky for the specific pn parameters.
        """

        #Distance vector from observer to emission point
        D = (rf-ri)
        Dnorm = norm(D)
        nD = D/Dnorm

        if pncor == False:
            return Dnorm , nD  
        else:
            #Angular momentum of photon orbit
            L     = cross(ri,rf)
            #Lnorm = norm(L)

            #Impact parameter
            b = cross(D , L)/Dnorm**2
            bnorm = norm(b)

            #Normalized vectors or observer and emitter
            rinorm = norm(ri)
            rfnorm = norm(rf)
            nri    = ri/rinorm
            nrf    = rf/rfnorm

            #0th order term (Euclidean)
            pn0 = nD

            #1st order corrections
            pn1_dx1_parallel        =  ( -(self.T1 + self.V1 + self.N1 + self.H1)/rfnorm + (self.N1/3) * (bnorm**2/rfnorm**3) ) * nD
            pn1_dx1_perpendicular   =  (  (self.T1 + self.V1)*( dot(nrf,nD) + 1 ) + (self.N1/3)*(dot(nrf,nD)**3 +1) )*b/bnorm**2

            pn1_x1_perpendicular    = -(1/Dnorm)*(  (self.T1 + self.V1 +self.N1/3)*( rfnorm - rinorm + Dnorm )/bnorm**2 + (self.N1/3)*(1/rfnorm - 1/rinorm) )*b
            pn1_x1_parallel_scalar  = -( -(self.T1 + self.V1 + self.N1 + self.H1)*np.log( (rfnorm/rinorm)*( dot(nrf,nD) + 1 )/( dot(nri,nD) + 1 )) + (self.N1/3) * ( dot(nrf,nD) - dot(nri,nD))  )

            #Light deflection vector at the observer position
            dxdtau    = pn0 + pn1_dx1_parallel + pn1_dx1_perpendicular + pn1_x1_perpendicular
            
            #Time interval between emission and absorption
            delta_tau = Dnorm + pn1_x1_parallel_scalar

            return delta_tau , dxdtau 

    def dAdr(self,r,v,sigma):    
        n = r/norm(r)
        t1 = (self.N1/3)*dot(v, cross(n,cross(sigma,n)))/norm(r)
        t2 = -(self.T1 + self.V1 + self.N1 + self.H1)* dot(v,sigma+n)/( norm(r) *(1+dot(n,sigma)))

        return t1+t2
    
    def dtdt0(self,ri,rf,vi,vf):
        
        D = (rf-ri)
        Dnorm = norm(D)
        nD = D/Dnorm

        return 1/(1 - dot(nD,vf-vi)+ self.m*(self.dAdr(rf,vf,nD) - self.dAdr(ri,vi,nD)) )  
        
        