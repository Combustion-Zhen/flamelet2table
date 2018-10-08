import numpy as np
import h5py
from scipy import interpolate

def tableLookUpTwoImplicit(
        x_ave, x_var, p0, p1,
        filename = 'flameletTable.h5',
        p0_name = 'ProgressVariable', 
        p1_name = 'ProgressVariableVariance'):

    with h5py.File(filename,'r') as f:
        table = f['flameletTable']

        variable = list(f['variable'])
        x_ave_axis = np.array(f[table.dims[1].label])
        x_var_axis = np.array(f[table.dims[2].label])
        param_axis_0 = np.array(f[table.dims[3].label])
        param_axis_1 = np.array(f[table.dims[4].label])

        data = np.array(table)

    idx_p0 = variable.index( p0_name )
    idx_p1 = variable.index( p1_name )

    param_table = np.empty((2,)+data.shape[3:])

    # interpolate with explicit index
    for i in range( param_axis_0.size ):
        for j in range( param_axis_1.size ):
            f = interpolate.interp2d(x_var_axis, x_ave_axis, 
                                     data[idx_p0,:,:,i,j] )
            param_table[0,i,j] = f(x_var, x_ave)

            f = interpolate.interp2d(x_var_axis, x_ave_axis,
                                     data[idx_p1,:,:,i,j] )
            param_table[1,i,j] = f(x_var, x_ave)

    # interpolate with implicit index
    # inverse table
    table_inverse = np.empty( (2, param_axis_0.size) )
    for i in range( param_axis_0.size ):
        f = interpolate.interp1d(
            param_table[1,i,:], param_table[0,i,:], bounds_error=False,
            fill_value=(param_table[0,i,0], param_table[0,i,-1]) )
        table_inverse[0,i] = f( p1 )

        f = interpolate.interp1d(
            param_table[1,i,:], param_axis_1, bounds_error=False,
            fill_value=(param_axis_1[0], param_axis_1[-1]))
        table_inverse[1,i] = f( p1 )

    f = interpolate.interp1d(
        table_inverse[0,:], table_inverse[1,:], bounds_error=False,
        fill_value=(table_inverse[1,0], table_inverse[1,-1]))
    param_1 = f( p0 )

    f = interpolate.interp1d(
        table_inverse[0,:], param_axis_0, bounds_error=False,
        fill_value=(param_axis_0[0], param_axis_0[-1]))
    param_0 = f( p0 )

    # interpolate all variables
    var_table = np.empty((data.shape[0],)+data.shape[3:])
    for k in range( data.shape[0] ):
        for i in range( param_axis_0.size ):
            for j in range( param_axis_1.size ):     
                f = interpolate.interp2d(x_var_axis, x_ave_axis, data[k,:,:,i,j])
                var_table[k,i,j] = f(x_var, x_ave)

    phi = np.zeros(var_table.shape[0])
    for k in range( var_table.shape[0] ):
        f = interpolate.interp2d( param_axis_1, param_axis_0, var_table[k,:,:] )
        phi[k] = f(param_0, param_1)

    return phi
