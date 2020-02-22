import glob
import argparse
import numpy as np
import h5py
from flamelet_integration import *
from beta_integration import delta_integration
from name_params import name2params

def single_param_table(
    mode = 'SLFM', dir_name = 'flamelets', output='flameletTable.h5',
    param_mesh = 'solution', param_pdf = 'delta',
    average_mesh = 'solution', average_num = 100,
    variance_mesh = 'geometric', variance_num = 15, variance_ratio = 1.1):

    if mode == 'SLFM' :
        independent_variable = 'Z'
        param_name = 'chi'
        ref_param = 'chi'
    elif mode == 'FPV' :
        independent_variable = 'Z'
        param_name = 'ProgressVariable'
        ref_param = 'chi'
    elif mode == 'FPI' :
        independent_variable = 'ProgressVariable'
        param_name = 'Z'
        ref_param = 'ProgressVariable'
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
    idx = np.argsort(param)

    param = param[idx]
    param = (param-param[0])/(param[-1]-param[0])

    filenames = np.array( filenames )[idx]

    # get the referecen flamelet
    flamelet = reference_solution(filenames, ref_param, p_str, p_end)

    # the variables to be integrated
    names = dependent_variable_names(
        flamelet, independent_variable)

    if param_pdf != 'delta' and mode != 'FPI' :
        names.append('{}Variance'.format(param_name))

    variable_names = np.array( names )

    names = dependent_variable_names_print(
        filenames[0], independent_variable)

    names_print = np.array( names )

    # the independent variable average axis
    independent_average = sequence_01(
        average_mesh, average_num, flamelet[independent_variable], 1.)

    # the variance axis
    normalized_variance = sequence_01(
        variance_mesh, variance_num, np.linspace(0.,1.), variance_ratio)

    # the parameter average axis
    param_average = sequence_01(
        param_mesh, average_num, param, 1.)

    # flamelet table with the parameter from solutions
    flamelet_table_solution = param_solution_integration(
        filenames, 
        independent_variable, independent_average, normalized_variance, 
        variable_names)

    flamelet_table = table_integration(flamelet_table_solution,
                                       param,
                                       param_average,
                                       normalized_variance,
                                       param_pdf)

    if mode == 'FPI' :
        # integration of the max progress variable for normalization
        normalization_param = np.zeros( (2, param.size) )
        for i in range( param.size ):
            params = name2params( filenames[i][p_str:p_end] )
            normalization_param[0,i] = params[ref_param]
        normalization_param[1,:] = np.square( normalization[0,:] )

        normalization_table = table_integration(normalization_param,
                                                param,
                                                param_average,
                                                normalized_variance,
                                                param_pdf)
    elif param_pdf != 'delta' :
        # variance for implicit parameter
        idx = list(variable_names).index( param_name )
        flamelet_table[-1,:,:,:,:] -= np.square( flamelet_table[idx,:,:,:,:] )

    # name of data axis
    axis = []
    axis.append( 'variables' )
    axis.append( '{}Average'.format(independent_variable) )
    axis.append( '{}NormalizedVariance'.format(independent_variable) )
    axis.append( 'Parameter{}Average'.format(param_name) )
    axis.append( 'Parameter{}NormalizedVariance'.format(param_name) )

    # save the flamelet table
    with h5py.File(output, 'w') as f:
        
        f['flameletTable'] = flamelet_table
        
        # strings
        dt = h5py.special_dtype(vlen=str)
        ds = f.create_dataset(axis[0],
                              names_print.shape,
                              dtype=dt)
        ds[...] = names_print

        f[axis[1]] = independent_average
        f[axis[2]] = normalized_variance
        f[axis[3]] = param_average

        if param_pdf != 'delta':
            f[axis[4]] = normalized_variance
        else:
            del axis[-1]

        for i, v in enumerate(axis):
            f['flameletTable'].dims[i].label = v
            f['flameletTable'].dims.create_scale(f[v], v)
            f['flameletTable'].dims[i].attach_scale(f[v])

        if mode == 'FPI' :
            f['maxProgressVariable'] = normalization_table

            f['maxProgressVariable'].dims[0].label = 'MeanAndSquareMean'
            for i, v in enumerate(axis[3:]):
                f['maxProgressVariable'].dims[i+1].label = v

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m', '--mode',
        default = 'FPV',
        type = str,
        help = 'use of the flamelet solutions: SLFM/[FPV]/FPI')

    parser.add_argument(
        '-f', '--folder',
        default = 'flamelets',
        type = str,
        help = 'folder of the flamelet solutions [flamelets]')

    parser.add_argument(
        '-o', '--output',
        default = 'flameletTable.h5',
        type = str,
        help = 'output file name [flameletTable.h5]')

    parser.add_argument(
        '-p', '--parameter-mesh',
        default = 'uniform',
        type = str,
        help = 'mesh of the flamelet parameter solution/[uniform]')

    parser.add_argument(
        '--parameter-pdf',
        default = 'delta',
        type = str,
        help = 'pdf of the flamelet parameter [delta]/beta')

    parser.add_argument(
        '-a', '--average-mesh',
        default = 'uniform',
        type = str,
        help = 'mesh of average solution/[uniform]')

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
        mode = args.mode, dir_name = args.folder, output = args.output,
        param_mesh = args.parameter_mesh, param_pdf = args.parameter_pdf,
        average_mesh = args.average_mesh, average_num = args.number_average,
        variance_mesh = args.variance_mesh, variance_num = args.number_variance,
        variance_ratio = args.ratio_variance)
