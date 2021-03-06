import numpy as np
from name_params import *
from beta_integration import *

def single_solution_integration(
        solution, x_name, x_ave, x_var, y_names):

    x = solution[x_name]

    f = np.empty((y_names.size, x.size))

    for i, name in enumerate(y_names):
        if name.endswith('Variance'):
            f[i,:] = np.square( solution[name[:-8]] )
        else:
            f[i,:] = solution[name]

    integration = beta_integration_table(f, x, x_ave, x_var)

    return integration

def param_solution_integration(
        filenames, x_name, x_ave, x_var, y_names):

    table = np.empty((y_names.size, x_ave.size, x_var.size, filenames.size))

    for i, filename in enumerate( filenames ):
        solution = np.genfromtxt(filename, names=True, delimiter=',')

        table[:,:,:,i] = single_solution_integration(
            solution, x_name, x_ave, x_var, y_names)

    return table


def table_integration( f, x, x_ave, x_var, pdf ):

    if pdf == 'beta' :
        table = beta_integration_table(f, x, x_ave, x_var)
    else:
        table = delta_integration(f, x, x_ave)

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

    return names

def dependent_variable_names_print(filename, independent_variable):
    with open(filename, 'r') as f:
        names = f.readline().rstrip().split(',')

    names.remove( independent_variable )

    return names

def sequence_01(mesh, npts, solution, ratio):

    if mesh == 'solution' :
        return solution
    elif mesh == 'geometric' :
        return geometric_progression_01(npts, ratio)
    else :
        return np.linspace(0., 1., num = npts)

def reference_solution(filenames, ref_param, p_str, p_end):

    ref_var = np.empty(filenames.size)

    for i, filename in enumerate(filenames):
        ref_var[i] = name2params(filename[p_str:p_end])[ref_param]

    filename = filenames[-1]
    for i in range( filenames.size-2 ):
        if ref_var[i] < ref_var[i+1] and ref_var[i+1] > ref_var[i+2]:
            filename = filenames[i+1]

    return np.genfromtxt(filename, names=True, delimiter=',')
