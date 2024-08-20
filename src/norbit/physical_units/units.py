import numpy as np

#Constants of nature
c = 299792458 #m/s
G = 6.67430e-11 #m/kg/s^2

#prefixes
deci  = 1e-1
centi = 1e-2
mili  = 1e-3
micro = 1e-6
nano  = 1e-9

kilo = 1e3
mega = 1e6
giga = 1e9

#Time quantities
second  = 1 
minute  = 60*second
hour    = 60*minute
day     = 24*hour 
week    = 7*day
month   = 30*day
year    = 365.25*day

#Lenght units
meter = 1
kilometer           = 1e3*meter

solar_radius        = 696340*kilometer
mercury_radius      = 2439.7*kilometer
venus_radius        = 6051.8*kilometer 
earth_radius        = 6371*kilometer 
moon_radius         = 1737.4*kilometer
mars_radius         = 3389.5*kilometer
jupiter_radius      = 69911*kilometer
saturn_radius       = 58232*kilometer
uranus_radius       = 25362*kilometer 
neptune_radius      = 24622*kilometer

astronomical_unit   = 149597870700*meter
parsec              = 3600*180/np.pi*astronomical_unit
megaparsec          = mega*parsec

#Mass units
kilogram = 1.0

solar_mass      = 1.98841e30*kilogram
mercury_mass    = 3.285e23*kilogram
venus_radius    = 4.867e24*kilogram
earth_mass      = 5.972e24*kilogram
moon_mass       = 7.3e22*kilogram
jupiter_mass    = 1.89813e27*kilogram
saturn_mass     = 5.683e26*kilogram
uranus_mass     = 1.024e24*kilogram
neptune_mass    = 8.681e25*kilogram

#Angular conversion factors
rad_to_as = (180.0/np.pi)*60*60
as_to_rad = 1.0/rad_to_as
deg_to_rad = np.pi/180.0
rad_to_deg = 1.0/deg_to_rad

#galactic center units
Mbh = 4.29701742727e6*solar_mass
R0  = 8277.09055007*parsec
Rg  = G*Mbh/c**2 