import numpy as np
from .vector_class import vec3

def dot(v1 , v2):
    """Returns the dot product of two vectors
    """
    if (not isinstance(v1,vec3) or not isinstance(v2,vec3)):
        raise ValueError('Both arguments must be vec3 class instances')
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z

def cross(v1 , v2):
    """Returns the cross product of two vectors
    """
    if (not isinstance(v1,vec3) or not isinstance(v2,vec3)):
        raise ValueError('Both arguments must be vec3 class instances')
        
    xc = v1.y * v2.z - v1.z * v2.y
    yc = v1.z * v2.x - v1.x * v2.z
    zc = v1.x * v2.y - v1.y * v2.x

    return vec3([xc,yc,zc])

def norm(v):
    """Returns the norm of a vector
    """
    if not isinstance(v,vec3):
        raise ValueError('Both arguments must be vec3 class instances')
    return np.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
