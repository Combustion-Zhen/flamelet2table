import numpy as np
from name_params import *
from beta_integration import beta_integration

def single_solution_integration(
        solution, x_name, x_ave, x_var, y_names):

    # flamelet: arrray, flamelet solution in structured numpy array

    integration = np.empty((x_var.size, x_ave.size, y_names.size))

    for i, var in enumerate(x_var):
        for j, ave in enumerate(x_ave):
            for k, name in enumerate(y_names):
                integration[i,j,k] = beta_integration(
                    solution[name], solution[x_name], ave, var)

    return integration

def param_solution_integration(
        filenames, x_name, x_ave, x_var, y_names):

    table = np.empty((filenames.size, x_var.size, x_ave.size, y_names.size))

    for i, filename in enumerate( filenames ):
        solution = np.genfromtxt(filename, names=True, delimiter=',')

        table[i,:,:,:] = single_solution_integration(
            solution, x_name, x_ave, x_var, y_names)

    return table

def param_beta_integration(solution, p_ave, p_var):

    table = np.empty((p_var.size, p_ave.size)+solution.shape[1:])

    return table

def geometric_progression_01( n, ratio ):

    v = np.geomspace( 1., np.power(ratio, n-2), num = n-1 )
    r = np.zeros( n )

    for i in range( n-1 ):
        r[i+1] = np.sum( v[:i+1] )
    r /= r[-1]

    return r

def dependent_variable_names(flamelet, independent_variable):

    names = list( flamelet.dtype.names )
    names.remove( independent_variable )

    return np.array( names )

def average_sequence(mesh, solution, npts):

    if mesh == 'solution' :
        return solution
    else :
        return np.linspace(0., 1., num = npts)

def sequence_01(mesh, npts, ratio):

    if mesh == 'geometric' :
        return geometric_progression_01(npts, ratio)
    else :
        return np.linspace(0., 1., num = npts)

def reference_solution(filenames, ref_param, p_str, p_end):

    ref_var = np.empty(filenames.size)

    for i, filename in enumerate(filenames):
        ref_var[i] = name2params(filename[p_str:p_end])[ref_param]

    filename = filenames[0]
    for i in range( filenames.size-2 ):
        if ref_var[i] < ref_var[i+1] and ref_var[i+1] > ref_var[i+2]:
            filename = filenames[i+1]

    return np.genfromtxt(filename, names=True, delimiter=',')
