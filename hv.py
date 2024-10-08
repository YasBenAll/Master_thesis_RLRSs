from pymoo.indicators.hv import HV
import numpy as np 
import math
from typing import List
from copy import deepcopy
def sparsity(front: List[np.ndarray]) -> float:
    """Sparsity metric from PGMORL.

    Basically, the sparsity is the average distance between each point in the front.

    Args:
        front: current pareto front to compute the sparsity on

    Returns:
        float: sparsity metric
    """
    if len(front) < 2:
        return 0.0

    sparsity_value = 0.0
    m = len(front[0])
    front = np.array(front)
    for dim in range(m):
        objs_i = np.sort(deepcopy(front.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity_value += np.square(objs_i[i] - objs_i[i - 1])
    sparsity_value /= len(front) - 1

    return sparsity_value


def hypervolume(ref_point, points):
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)

ref_point = np.array([0.0, 0.0])

degrees = [0, 45, 90]
# define a set of identically distributed points between 0 and 90 degrees
degrees = np.linspace(0, 90, 3)


points = np.array([[math.sin(degree*math.pi/180), math.cos(degree*math.pi/180)]for degree in degrees])

print(hypervolume(ref_point, points))
print(math.pi*1/4)

print(sparsity(points))