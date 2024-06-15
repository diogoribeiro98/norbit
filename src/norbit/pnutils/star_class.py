import numpy as np
from ..vector import vec3
from ..vector import cross
from ..physical_units import units
from .star_data import star_data
from .orbital_elements import get_angular_momentum_vector, get_apocenter_unit_vectors

class star:

    def __init__(self, name, dictionary = True):

        if (star_data.get(name) == None) and (dictionary == True):
            error_message = "No star with name {} in the dictionary.".format(name)
            error_message += " Avaliable names are {}".format(list(star_data.keys()))
            raise ValueError(error_message)
        else:

            #GC quantities
            self.Rg  = units.Rg/units.astronomical_unit #in AU
            self.dgc = units.R0/units.astronomical_unit #in AU
            self.rad_to_as = (180/np.pi)*60*60
            self.as_to_rad = 1/self.rad_to_as
            self.deg_to_rad = np.pi/180
            self.rad_to_deg = 1/self.deg_to_rad
            
            #Load star orbital elements
            sdata = star_data[name]   

            self.name   = name
            self.Omega  = sdata['OmegaCapital']*self.deg_to_rad
            self.inc    = sdata['i']*self.deg_to_rad
            self.omega  = sdata['omega']*self.deg_to_rad
            self.a      = sdata['a']/self.rad_to_as*(self.dgc/self.Rg)
            self.e      = sdata['e']

            #Date at apogee
            self.Ta = sdata['Tp'] + sdata['T']/2
            self.T  = sdata['T']
            
            #Perigee and apogee
            self.ra = self.a*(1+self.e)
            self.rp = self.a*(1-self.e)

            #Velocities at apogee and perigee
            self.va = np.sqrt(2/self.ra - 1/self.a)
            self.vp = np.sqrt(2/self.rp - 1/self.a)

            #Angular momentum vector of the orbit
            self.L_vec = get_angular_momentum_vector(self.Omega, self.inc)

            #Perigee and apogee position vectors
            nr_apo, nv_apo = get_apocenter_unit_vectors(self.Omega, self.inc, self.omega)

            self.ra_vec = self.ra*nr_apo
            self.rp_vec = -self.rp*nr_apo

            #Velocity at apogee
            self.va_vec = self.va*nv_apo

    def info(self):
        print("S2 orbital parameters")
        print("Semi-major axis (a) : " , self.a  , "Rg")
        print("Perigee (rp) : " , self.rp , "Rg")
        print("Apogee (ra) : " , self.ra , "Rg")
        print("Excentricity (e) : " , self.e  )
        print("Inclination (inc) : " , self.inc/self.deg_to_rad , "degrees"  )
        print("Longitude of the ascending node (Omega) : " , self.Omega/self.deg_to_rad, "degrees"  )
        print("Argument of perihastron (omg) : " , self.omega/self.deg_to_rad , "degrees"  )
