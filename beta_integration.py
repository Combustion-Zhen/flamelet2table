import numpy as np
import numpy.polynomial.polynomial as P
from scipy.stats import beta
from scipy.interpolate import interp1d
import scipy as sp

def beta_coef(ave, var):

    a = ave*(1./var-1.)
    b = (1.-ave)*(1./var-1.)

    return a, b

def beta_integration_analytic(f, x, a, b):
    
    rv0 = beta(a, b)
    cdf0 = rv0.cdf(x)
    B0 = sp.special.beta(a, b)
    
    rv1 = beta(a+1., b)
    cdf1 = rv1.cdf(x)
    B1 = sp.special.beta(a+1, b)
    
    c0 = np.zeros(x.size)
    c1 = np.zeros(x.size)
    
    for i in range(x.size-1):
        c = P.polyfit(x[i:i+2], f[i:i+2], 1)
        
        c0[i] -= c[0]
        c0[i+1] += c[0]
        
        c1[i] -= c[1]
        c1[i+1] += c[1]
        
    c1 *= B1/B0
    
    return np.sum(c0*cdf0+c1*cdf1)


def delta_integration(f, x, x_ave):

    y = interp1d(x, f, kind='cubic', axis=0)

    return y(x_ave)


def bimodal_integration(f, x_ave):

    return f[0]*(1.-x_ave)+f[-1]*x_ave


def beta_integration(f, x, x_ave, x_nvar):
    
    epsilon = 1.e-9
    
    if x_ave < epsilon:
        return f[0]
    elif x_ave > 1.-epsilon:
        return f[-1]
    elif x_nvar < epsilon:
        return delta_integration(f, x, x_ave)
    elif x_nvar > 1.-epsilon:
        return bimodal_integration(f, x_ave)
    else:
        a, b = beta_coef(x_ave, x_nvar)
        return beta_integration_analytic(f, x, a, b)
