from pymoo.indicators.hv import HV
import numpy as np 
import math

def hypervolume(ref_point, points):
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)

ref_point = np.array([0.0, 0.0])

degrees = [0, 45, 90]
# define a set of identically distributed points between 0 and 90 degrees
degrees = np.linspace(0, 90, 100)


points = np.array([[math.sin(degree*math.pi/180), math.cos(degree*math.pi/180)]for degree in degrees])

print(hypervolume(ref_point, points))
print(math.pi*1/4)