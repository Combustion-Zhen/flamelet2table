"""
A python script to transfer FlameMaster output to csv tables
"""

import numpy as np
import os
import sys
import glob
import argparse
from name_params import params2name

def FM2csv(mode = 'SLFM', 
           flamelet_prefix = 'CH4_p01_0',
           FlameMaster_dir = 'OutSteady', 
           csv_dir = 'flamelets'):

    if not os.path.exists(FlameMaster_dir):
        sys.exit('FlameMaster output not found')

    os.makedirs(csv_dir,exist_ok=True)

    # the number of columns in FlameMaster output
    NCOL = 5
    file_suffix = 'csv'

    # read the species names into dictionary
    # key: species names in FlameMaster
    # value: species names in Chemkin mechanism
    name_dict = {}
    with open('{}/speciestranslated'.format(FlameMaster_dir),'r') as f:
        for line in f:
            names = line.split()
            name_dict['-'.join(['massfraction',names[1]])] = names[0] 

    name_dict['temperature'] = 'T'
    name_dict['density'] = 'rho'

    extra_var = ['Z', 'chi', 'lambda', 'cp', 'mu', 'D',
                 'ProdRateCO2', 'ProdRateH2O', 'ProdRateCO', 'ProdRateH2',
                 'TotalEnthalpy', 'HeatRelease']

    for name in extra_var:
        name_dict[name] = name

    for flamelet in glob.glob('{0}/{1}*'.format(FlameMaster_dir, 
                                                flamelet_prefix)):

        # skip diverged solution
        if flamelet.endswith('noC'):
            continue

        params = {}

        params['chi'] = float(
            flamelet[flamelet.find('chi')+3:flamelet.find('tf')])

        if mode == 'FPV' :
            params['T'] = float(
                flamelet[flamelet.find('Tst')+3:])

        file_prefix = params2name( params )

        with open(flamelet,'r') as f:

            # read the header part
            for line in f:
                if line.startswith('gridPoints'):
                    npts = int(line.split()[-1])
                    nrow = np.ceil(npts/NCOL)
                elif line.startswith('body'):
                    break

            name_FlameMaster = list(name_dict.keys())
            name_csv = list(name_dict.values())

            data = np.empty((npts, len(name_FlameMaster)),order='F')

            for line in f:

                if line.strip() == '':
                    continue
                elif line.startswith('trailer'):
                    break

                var_name = line.split()[0]

                # read data
                var = []
                for i in np.arange(nrow):
                    data_line = f.readline().split()
                    var.extend([float(x) for x in data_line])

                # map names
                if var_name in name_FlameMaster :
                    i = name_csv.index( name_dict[var_name] )
                    data[:,i] = np.array( var )

            # calculate Diffusivity with Le = 1
            idx_D = name_csv.index( 'D' )
            idx_lambda = name_csv.index( 'lambda' )
            idx_rho = name_csv.index( 'rho' )
            idx_cp = name_csv.index( 'cp' )
            data[:,idx_D] = data[:,idx_lambda]/(data[:,idx_rho]*data[:,idx_cp])

            np.savetxt('{}/{}.{}'.format(csv_dir,file_prefix,file_suffix),
                       data, 
                       fmt = '%12.6e', 
                       delimiter = ',', 
                       header = ','.join(name_csv),
                       comments='')

    return

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--mode',
        default = 'SLFM',
        type = str,
        help = 'use of the flamelet solutions: [SLFM]/FPV')
    parser.add_argument(
        '-p', '--prefix',
        default = 'CH4_p01_0',
        type = str,
        help = 'file prefix of the flamelet solution [CH4_p01_0]')
    parser.add_argument(
        '--FlameMasterFolder',
        default = 'OutSteady',
        type = str,
        help = 'folder of the FlameMaster solutions [OutSteady]')
    parser.add_argument(
        '--csvFolder',
        default = 'flamelets',
        type = str,
        help = 'folder for output csv files [flamelets]')
    args = parser.parse_args()

    FM2csv(mode = args.mode,
           flamelet_prefix = args.prefix,
           FlameMaster_dir = args.FlameMasterFolder, 
           csv_dir = args.csvFolder)
