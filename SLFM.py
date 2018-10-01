import os
import glob
import argparse
import numpy as np
import h5py
from beta_integration import beta_integration
from flamelet_integration import *
from name_params import *

def table_SLFM(dir_name = 'flamelets',
               average_distribution = 'solution',
               average_num = 100,
               variance_distribution = 'geometric',
               variance_num = 15,
               variance_ratio = 1.1):

    independent_variable = 'Z'

    # get the flamelet solutions
    file_suffix = 'csv'

    chi = np.zeros(1)
    os.chdir(dir_name)
    for filename in glob.glob('.'.join(['*', file_suffix])):
        params = name2params( filename[:-1-len(file_suffix)] )
        chi = np.append(chi, params['chi'])
    os.chdir('..')
    chi = np.delete( chi, 0, 0 )
    chi = np.sort( chi )

    # take the flamelet solution with largest chi_st
    params = { 'chi' : chi[-1] }
    file_prefix = params2name( params )

    filename = '{0}/{1}.{2}'.format(dir_name, file_prefix, file_suffix)
    flamelet = np.genfromtxt(filename, names=True, delimiter=',')

    # the variables to be integrated
    variable_names = flamelet_dependent_variable(flamelet, independent_variable)

    # the average axis
    independent_average = independent_variable_average(
        flamelet,independent_variable,average_distribution,average_num)

    # the variance axis
    independent_variance = variance_series(
        variance_distribution, variance_num, variance_ratio)

    flamelet_table = np.empty((chi.size, 
                               independent_variance.size, 
                               independent_average.size, 
                               variable_names.size))

    for l, chi_st in enumerate(chi):
        params = { 'chi' : chi_st }
        file_prefix = params2name( params )

        filename = '{0}/{1}.{2}'.format(dir_name, file_prefix, file_suffix)
        flamelet = np.genfromtxt(filename, names=True, delimiter=',')    

        flamelet_table[l,:,:,:] = single_flamelet_integration(
            flamelet,
            independent_variable, 
            independent_average, 
            independent_variance, 
            variable_names)

    # save the flamelet table
    with h5py.File('flameletTable.h5', 'w') as f:
        
        f['flameletTable'] = flamelet_table
        
        # strings
        dt = h5py.special_dtype(vlen=str)
        ds = f.create_dataset('variable',
                              variable_names.shape,
                              dtype=dt)
        ds[...] = variable_names
        
        f['mixtureFractionAverage'] = independent_average
        f['mixtureFractionNormalizedVariance'] = independent_variance
        f['stoichiometricScalarDissipationRate'] = chi
        
        f['flameletTable'].dims.create_scale(
                f['variable'],
                'variable')
        
        f['flameletTable'].dims.create_scale(
                f['mixtureFractionAverage'],
                'mixtureFractionAverage')
        
        f['flameletTable'].dims.create_scale(
                f['mixtureFractionNormalizedVariance'],
                'mixtureFractionNormalizedVariance')
        
        f['flameletTable'].dims.create_scale(
                f['stoichiometricScalarDissipationRate'],
                'stoichiometricScalarDissipationRate')
        
        f['flameletTable'].dims[0].attach_scale(
                f['stoichiometricScalarDissipationRate'])

        f['flameletTable'].dims[1].attach_scale(
                f['mixtureFractionNormalizedVariance'])

        f['flameletTable'].dims[2].attach_scale(
                f['mixtureFractionAverage'])

        f['flameletTable'].dims[3].attach_scale(
                f['variable'])

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--folder',
        default = 'flamelets',
        type = str,
        help = 'folder of the flamelet solutions')

    parser.add_argument(
        '-a', '--average-distribution',
        default = 'solution',
        type = str,
        help = 'mesh of average')

    parser.add_argument(
        '--number-average',
        default = 100,
        type = int,
        help = 'the number of points on the axis of average')

    parser.add_argument(
        '-v', '--variance-distribution',
        default = 'geometric',
        type = str,
        help = 'mesh of variance')

    parser.add_argument(
        '--number-variance',
        default = 15,
        type = int,
        help = 'the number of points on the axis of variance')

    parser.add_argument(
        '--ratio-variance',
        default = 1.1,
        type = float,
        help = 'growth rate of the variance mesh')

    args = parser.parse_args()

    table_SLFM(dir_name = args.folder,
               average_distribution = args.average_distribution,
               average_num = args.number_average,
               variance_distribution = args.variance_distribution,
               variance_num = args.number_variance,
               variance_ratio = args.ratio_variance)
