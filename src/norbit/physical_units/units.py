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
kilometer           = 1e4*meter

parsec              = 3.08567758149e16*meter
kiloparsec          = kilo*parsec
megaparsec          = mega*parsec

astronomical_unit   = 149597870700*meter

#Mass units
kilogram   = 1.0
solar_mass = 1.98841e30*kilogram

#Angular conversion factors
rad_to_as = (180.0/np.pi)*60*60
as_to_rad = 1.0/rad_to_as
deg_to_rad = np.pi/180.0
rad_to_deg = 1.0/deg_to_rad

#galactic center units
Mbh = 4.29701742727e6*solar_mass
R0  = 8277.09055007*parsec
Rg  = G*Mbh/c**2 