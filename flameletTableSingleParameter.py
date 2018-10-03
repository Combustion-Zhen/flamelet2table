import glob
import argparse
import numpy as np
import h5py
from flamelet_integration import *
from beta_integration import delta_integration
from name_params import name2params

def single_param_table(
    mode = 'SLFM', dir_name = 'flamelets',
    param_mesh = 'solution', param_pdf = 'delta',
    average_mesh = 'solution', average_num = 100,
    variance_mesh = 'geometric', variance_num = 15, variance_ratio = 1.1):

    if mode == 'SLFM' :
        independent_variable = 'Z'
        param_name = 'chi'
        ref_param = 'chi'
    elif mode == 'FPV' :
        independent_variable = 'Z'
        param_name = 'T'
        ref_param = 'chi'
    else :
        print('mode not implemented')
        return

    # get the flamelet solutions
    file_suffix = 'csv'

    p_str = 1 + len( dir_name )
    p_end = -1 - len( file_suffix )

    # get file and parameter list
    param = []
    filenames = []

    for filename in glob.glob('{}/*.{}'.format(dir_name,file_suffix)):
        params = name2params(filename[p_str:p_end])
        param.append( params[param_name] )
        filenames.append( filename )

    param = np.array(param)
    idx = np.argsort(param)[::-1]

    param = param[idx]
    param = (param-param[-1])/(param[0]-param[-1])

    filenames = np.array( filenames )[idx]

    # get the referecen flamelet
    flamelet = reference_solution(filenames, ref_param, p_str, p_end)

    # the variables to be integrated
    variable_names = dependent_variable_names(flamelet, independent_variable)

    # the independent variable average axis
    independent_average = average_sequence(
        average_mesh, flamelet[independent_variable], average_num)

    # the variance axis
    normalized_variance = sequence_01(
        variance_mesh, variance_num, variance_ratio)

    # the parameter average axis
    param_average = average_sequence(param_mesh, param, average_num)

    # flamelet table with the parameter from solutions
    flamelet_table_solution = param_solution_integration(
        filenames, 
        independent_variable, independent_average, normalized_variance, 
        variable_names)

    # integration with repect to the parameter
    if param_mesh == 'beta':
        flamelet_table = param_beta_integration(
            flamelet_table_solution, param_average, normalized_variance)
        # variance integration to be implemented
    elif param_mesh != 'solution':
        flamelet_table = delta_integration(
            flamelet_table_solution, param, param_average)
    else:
        flamelet_table = flamelet_table_solution

    # name of data axis
    axis = []
    axis.append( '{}NormalizedVariance'.format(param_name) )
    axis.append( '{}Average'.format(param_name) )
    axis.append( '{}NormalizedVariance'.format(independent_variable) )
    axis.append( '{}Average'.format(independent_variable) )
    axis.append( 'variable' )

    # save the flamelet table
    with h5py.File('flameletTable.h5', 'w') as f:
        
        f['flameletTable'] = flamelet_table
        
        f[axis[1]] = param_average
        f[axis[2]] = normalized_variance
        f[axis[3]] = independent_average

        if param_pdf != 'delta':
            f[axis[0]] = normalized_variance
        else:
            del axis[0]
        
        # strings
        dt = h5py.special_dtype(vlen=str)
        ds = f.create_dataset(axis[-1],
                              variable_names.shape,
                              dtype=dt)
        ds[...] = variable_names

        for i, v in enumerate(axis):
            f['flameletTable'].dims.create_scale(f[v], v)
            f['flameletTable'].dims[i].attach_scale(f[v])

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m', '--mode',
        default = 'SLFM',
        type = str,
        help = 'use of the flamelet solutions: [SLFM]/FPV')

    parser.add_argument(
        '-f', '--folder',
        default = 'flamelets',
        type = str,
        help = 'folder of the flamelet solutions [flamelets]')

    parser.add_argument(
        '-p', '--parameter-mesh',
        default = 'solution',
        type = str,
        help = 'mesh of the flamelet parameter [solution]/uniform')

    parser.add_argument(
        '--parameter-pdf',
        default = 'delta',
        type = str,
        help = 'pdf of the flamelet parameter [delta]')

    parser.add_argument(
        '-a', '--average-mesh',
        default = 'solution',
        type = str,
        help = 'mesh of average [solution]/uniform')

    parser.add_argument(
        '--number-average',
        default = 100,
        type = int,
        help = 'the number of points on the axis of average [100]')

    parser.add_argument(
        '-v', '--variance-mesh',
        default = 'geometric',
        type = str,
        help = 'mesh of variance [geometric]/uniform')

    parser.add_argument(
        '--number-variance',
        default = 15,
        type = int,
        help = 'the number of points on the axis of variance [15]')

    parser.add_argument(
        '--ratio-variance',
        default = 1.1,
        type = float,
        help = 'growth rate of the variance mesh [1.1]')

    args = parser.parse_args()

    args = parser.parse_args()

    single_param_table(
        mode = args.mode, dir_name = args.folder,
        param_mesh = args.parameter_mesh, param_pdf = args.parameter_pdf,
        average_mesh = args.average_mesh, average_num = args.number_average,
        variance_mesh = args.variance_mesh, variance_num = args.number_variance,
        variance_ratio = args.ratio_variance)
