import numpy as np
from beta_integration import beta_integration

def single_flamelet_integration(
        flamelet,
        independent_name, independent_average, independent_variance,
        dependent_names):

    # flamelet: arrray, flamelet solution in structured numpy array

    flamelet_integration = np.empty(
        (independent_variance.size, 
         independent_average.size,
         dependent_names.size))

    for i, var in enumerate(independent_variance):
        for j, ave in enumerate(independent_average):
            for k, name in enumerate(dependent_names):
                flamelet_integration[i,j,k] = beta_integration(
                    flamelet[name], flamelet[independent_name], ave, var)

    return flamelet_integration

def geometric_progression_01( n, ratio ):

    v = np.geomspace( 1., np.power(ratio, n-2), num = n-1 )
    r = np.zeros( n )

    for i in range( n-1 ):
        r[i+1] = np.sum( v[:i+1] )
    r /= r[-1]

    return r
