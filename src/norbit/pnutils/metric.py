import numpy as np

class metric:
    """A class for defining GR metrics in cartesian form.
    """
    def __init__(self,
                 A,B,D,
                 pn_coefficients_1st_order,
                 pn_coefficients_2nd_order,
                 pn_coefficients_3rd_order
                 ):

        #Metric functions
        self.A = A
        self.B = B
        self.D = D

        #Post newtonian coefficients
        self.pn_coefficients_1st_order = np.array(pn_coefficients_1st_order)
        self.pn_coefficients_2nd_order = np.array(pn_coefficients_2nd_order)
        self.pn_coefficients_3rd_order = np.array(pn_coefficients_3rd_order)

    def redshift_factor(self, r_vec, v_vec, gr_effects=True):
        """Calculate redshift factor associated with a specific position and velocity of the emitter

        Args:
            r_vec (vec3): 3d vector representing the position in space
            v_vec (vec3): 3d vector representing the velocity of the emitter
            gr_effects (bool, optional): If False, only special relativity effects are calculated. Defaults to True.

        Returns:
            _type_: _description_
        """
  
        r  = np.linalg.norm(r_vec)
        v  = np.linalg.norm(v_vec)

        nr = r_vec/np.linalg.norm(r_vec) 

        if gr_effects==False:
            return -minkowsky_metric.A(r) + v**2*minkowsky_metric.B(r) + np.dot(nr,v_vec)**2*minkowsky_metric.D(r)
        else:
            return -self.A(r) + v**2*self.B(r) + np.dot(nr,v_vec)**2*self.D(r)

#Standard metrics
minkowsky_metric = metric(
    A = lambda x: 1, 
    B = lambda x: 1, 
    D = lambda x: 1,
    pn_coefficients_1st_order = [0,0,0,0],
    pn_coefficients_2nd_order = [0,0,0,0],
    pn_coefficients_3rd_order = [0,0,0,0],
    ) 

schwarzschild_metric = metric(
    A = lambda x: 1-2/x, 
    B = lambda x: 1, 
    D = lambda x: (2/x)/(1-2/x),
    pn_coefficients_1st_order = [-1,-2, 3, 2],
    pn_coefficients_2nd_order = [ 2, 0, 2, 4],
    pn_coefficients_3rd_order = [ 0, 0, 4, 8],
    )

