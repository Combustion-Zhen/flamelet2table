import numpy as np
import numpy.polynomial.polynomial as P
from scipy.stats import beta
from scipy.interpolate import interp1d
import scipy as sp

# calculate the beta pdf coefficients alpha and beta
def beta_coef(ave, var):

    a = ave*(1./var-1.)
    b = (1.-ave)*(1./var-1.)

    return a, b


# analytic integration with beta pdf, with linear fit of the target function
def beta_integration_analytic(f, x, B, CDF0, CDF1):

    c0 = np.zeros(x.size)
    c1 = np.zeros(x.size)

    for i in range(x.size-1):
        cl1 = (f[i+1] - f[i])/(x[i+1]-x[i])
        cl0 = f[i] - cl1*x[i]

        c0[i] -= cl0
        c0[i+1] += cl0

        c1[i] -= cl1
        c1[i+1] += cl1

    c1 *= B

    return np.sum(c0*CDF0+c1*CDF1)


# integration with the delta pdf, implemented as a linear interp
def delta_integration(f, x, x_ave):

    y = interp1d(x, f, kind='linear')

    return y(x_ave)


# integration with the bimodal pdf
def bimodal_integration(f, x_ave):

    return f[0]*(1.-x_ave)+f[-1]*x_ave


# beta integration of one average and one variance
def beta_integration(
        f, x, x_ave, x_nvar,
        B, CDF0, CDF1, EPS):

    if x_ave < EPS:
        return f[0]
    elif x_ave > 1.-EPS:
        return f[-1]
    elif x_nvar < EPS:
        return delta_integration(f, x, x_ave)
    elif x_nvar > 1.-EPS:
        return bimodal_integration(f, x_ave)
    else:
        return beta_integration_analytic(
                f, x, B, CDF0, CDF1)


# calculate the coefficients for analytic beta integration
def beta_integration_coef(x, x_ave, x_nvar):

    a, b = beta_coef(x_ave, x_nvar)

    rv0 = beta(a, b)
    cdf0 = rv0.cdf(x)
    B0 = sp.special.beta(a, b)

    rv1 = beta(a+1., b)
    cdf1 = rv1.cdf(x)
    B1 = sp.special.beta(a+1, b)

    B = B1/B0

    return B, cdf0, cdf1


# calculate the coefficients for analytic beta integration
def beta_integration_coef_table(x, x_ave, x_var, EPS):

    B = np.empty((x_ave.size, x_var.size))
    CDF0 = np.empty((x_ave.size, x_var.size, x.size))
    CDF1 = np.empty((x_ave.size, x_var.size, x.size))

    for j, ave in enumerate(x_ave):
        for k, var in enumerate(x_var):
            if ave > EPS and ave < 1.-EPS and var > EPS and var < 1.-EPS :
                B[j,k], CDF0[j,k,:], CDF1[j,k,:] = beta_integration_coef(
                    x, ave, var)

    return B, CDF0, CDF1


# beta integration of a average and variance table
def beta_integration_table(f, x, x_ave, x_var, EPS=1.e-9):

    f_flatten = np.reshape(f, (-1, x.size), order='F')

    variable_number = f_flatten.shape[0]

    table_flatten = np.empty((variable_number, x_ave.size, x_var.size))

    B, CDF0, CDF1 = beta_integration_coef_table(x, x_ave, x_var, EPS)

    for i, v in enumerate(f_flatten):
        for j, ave in enumerate(x_ave):
            for k, var in enumerate(x_var):
                table_flatten[i,j,k] = beta_integration(
                        v, x, ave, var,
                        B[j,k], CDF0[j,k,:], CDF1[j,k,:], EPS)

    table = np.reshape(table_flatten, 
                       f.shape[:-1]+(x_ave.size, x_var.size), 
                       order='F')

    return table
