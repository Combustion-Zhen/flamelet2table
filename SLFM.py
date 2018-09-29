import os
import glob
import argparse
import numpy as np
import h5py
from beta_integration import beta_integration
from name_params import *

def SLFM( dir_name = 'flamelets', n_Z_variance = 21 ):

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
    n_chi = chi.size

    # take the flamelet solution with largest chi_st
    params = { 'chi' : chi[-1] }
    file_prefix = params2name( params )

    filename = '{0}/{1}.{2}'.format(dir_name, file_prefix, file_suffix)
    flamelet = np.genfromtxt(filename, names=True, delimiter=',')

    # the number of variables to be integrated
    names = list(flamelet.dtype.names)
    names.remove('Z')
    variable_names = np.array( names )
    n_variable = variable_names.size

    # the Z_average axis
    Z_average = flamelet['Z']
    n_Z_average = Z_average.size

    # the Z_variance axis
    Z_variance = np.linspace(0., 1., num=n_Z_variance)

    flamelet_table = np.empty((n_chi, n_Z_variance, n_Z_average, n_variable))

    for l, chi_st in enumerate(chi):
        params = { 'chi' : chi_st }
        file_prefix = params2name( params )

        filename = '{0}/{1}.{2}'.format(dir_name, file_prefix, file_suffix)
        flamelet = np.genfromtxt(filename, names=True, delimiter=',')    
        for i, nvar in enumerate(Z_variance):
            for j, ave in enumerate(Z_average):
                for k, name in enumerate(variable_names):
                    flamelet_table[l,i,j,k] = beta_integration(
                            flamelet[name], flamelet['Z'],
                            ave, nvar)

    # save the flamelet table
    with h5py.File('flameletTable.h5', 'w') as f:
        
        f['flameletTable'] = flamelet_table
        
        # strings
        dt = h5py.special_dtype(vlen=str)
        ds = f.create_dataset('variable',
                              variable_names.shape,
                              dtype=dt)
        ds[...] = variable_names
        
        f['mixtureFractionAverage'] = Z_average
        f['mixtureFractionNormalizedVariance'] = Z_variance
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
        default = 21,
        type = int,
        help = 'the number of points on the axis of variance')
    args = parser.parse_args()

    SLFM(dir_name = args.folder,
         n_Z_variance = args.number)
