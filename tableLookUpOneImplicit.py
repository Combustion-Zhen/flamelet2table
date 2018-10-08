import numpy as np
import h5py
from scipy import interpolate

def tableLookUpOneImplicit(
        x_ave, x_var, p,
        filename = 'flameletTable.h5',
        p_name = 'ProgressVariable'):

    with h5py.File(filename,'r') as f:
        table = f['flameletTable']

        variable = list(f['variable'])
        x_ave_axis = np.array(f[table.dims[1].label])
        x_var_axis = np.array(f[table.dims[2].label])
        param_axis = np.array(f[table.dims[3].label])

        data = np.array(table)

    idx_p = variable.index( p_name )

    param_table = np.empty(data.shape[3])

    # interpolate with explicit index
    for i in range( param_axis.size ):
        f = interpolate.interp2d(x_var_axis, x_ave_axis, 
                                 data[idx_p,:,:,i] )
        param_table[i] = f(x_var, x_ave)

    # interpolate with implicit index
    # inverse table
    f = interpolate.interp1d(
        param_table, param_axis, bounds_error=False,
        fill_value=(param_axis[0], param_axis[-1]))
    param = f( p )

    # interpolate all variables
    var_table = np.empty((data.shape[0],data.shape[3]))
    for i in range( data.shape[0] ):
        for j in range( param_axis.size ):
            f = interpolate.interp2d(x_var_axis, x_ave_axis, data[i,:,:,j])
            var_table[i,j] = f(x_var, x_ave)

    f = interpolate.interp1d( param_axis, var_table )
    phi = f(param)

    return phi
