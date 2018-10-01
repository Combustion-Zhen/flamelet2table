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
    names = list(flamelet.dtype.names)
    names.remove(independent_variable)
    variable_names = np.array( names )

    # the average axis
    if average_distribution == 'solution' :
        independent_average = flamelet[independent_variable]
    else :
        independent_average = np.linspace(0., 1., num = average_num)

    # the variance axis
    if variance_distribution == 'geometric' :
        independent_variance = geometric_progression_01(variance_num,
                                                        variance_ratio)
    else :
        independent_variance = np.linspace(0., 1., num=variance_num)

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
        nargs = '?',
        const = 'flamelets',
        default = 'flamelets',
        type = str,
        help = 'folder of the flamelet solutions')

    parser.add_argument(
        '-n', '--number',
        nargs = '?',
        const = 2,
        default = 15,
        type = int,
        help = 'the number of points on the axis of variance')
    args = parser.parse_args()

    table_SLFM(dir_name = args.folder,
               variance_num = args.number)
