import numpy as np 

class vec3:
    """3D vector class suporting basic operations

    Operations:
        - addition
        - subtraction
        - scalar multiplication
        - scalar division
    """
    #Constructor
    def __init__(self, input_vector):
        """Constructor for vector class

        Args:
            input_vector (array): array with x,y,z components
        """
        self.x = input_vector[0]
        self.y = input_vector[1]
        self.z = input_vector[2]
        self.values = [self.x , self.y , self.z]

    #Useful functions
    def norm(self):
        """ Returns the euclidean norm of the vector
        
        Returns:
            float : norm of the vector
        """
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def rotate_along(self, vector, theta):
        """ Rotates the vector perpendicularly to the input vector according to the right hand rule using Rodrigues' rule. 
        See 'https://mathworld.wolfram.com/RodriguesRotationFormula.html'

        Args:
            vector vec3: Axis vector
            theta float: Rotation angle in radians

        Returns:
            vec3 : rotated vector
        """
        w = vector/vector.norm()

        c = np.cos(theta)
        s = np.sin(theta)

        #Define the rotation matrix
        mat = [ 
            [      c + w.x*w.x*(1-c) , -w.z*s + w.y*w.x*(1-c)  ,  w.y*s + w.z*w.x*(1-c) ] ,
            [  w.z*s + w.x*w.y*(1-c) ,      c + w.y*w.y*(1-c)  , -w.x*s + w.z*w.y*(1-c)],
            [ -w.y*s + w.x*w.z*(1-c) ,  w.x*s + w.y*w.z*(1-c)  ,      c + w.z*w.z*(1-c)]
        ]

        out_vec = [
            self.x*mat[0][0] + self.y*mat[0][1] + self.z*mat[0][2] ,
            self.x*mat[1][0] + self.y*mat[1][1] + self.z*mat[1][2] ,
            self.x*mat[2][0] + self.y*mat[2][1] + self.z*mat[2][2] 
        ]

        return self.__class__(out_vec)
    
    #Overwrite traditional functionalities
    def __add__(self, other):
        if isinstance(other, vec3):
            out_vec = tuple( a + b for a, b in zip(self.values, other.values) )
        elif isinstance(other, (int, float)):
            out_vec = tuple( a + other for a in self .values)
        else:
            raise ValueError("Addition with type {} not supported".format(type(other)))

        return self.__class__(out_vec)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, vec3):
            out_vec = tuple( a - b for a, b in zip(self.values, other.values) )
        elif isinstance(other, (int, float)):
            out_vec = tuple( a - other for a in self .values)
        else:
            raise ValueError("Addition with type {} not supported".format(type(other)))

        return self.__class__(out_vec)

    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self,other):
        if isinstance(other,vec3):
            out_vec = tuple( a*b for a, b in zip(self.values, other.values) )
        elif isinstance(other, (int, float)):
            out_vec = tuple( a * other for a in self.values )
        return self.__class__(out_vec)

    def __rmul__(self, other):
        return self.__mul__(other)    

    def __truediv__(self, other):
        if isinstance(other, vec3):
            out_vec = tuple( a/b for a, b in zip(self.values, other.values) )
        elif isinstance(other, (int, float)):
            out_vec = tuple( a / other for a in self.values ) 
        else:
            raise ValueError("Division with type {} not supported".format(type(other)))
        
        return self.__class__(out_vec)

    def __str__(self):
        return "[ {:.20f} , {:.20f} , {:.20f} ]".format(self.x , self.y, self.z)
