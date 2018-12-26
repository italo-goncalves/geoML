__all__ = ['cubic_conv_1D']

import numpy as np
#import scipy.sparse

def cubic_conv_1D(x, xnew):
    """
    Generates a sparse matrix for interpolating from a regular grid to
    a new set of positions in one dimension.
    
    Parameters
    ----------
    x : array of regularly spaced values.
    xnew : positions to receive the interpolated values. Its limits must be 
    confined within the limits of x.
    
    Returns
    -------
    W : a sparse matrix
    """
    if np.std(np.diff(x)) >  1e-9:
        raise Exception("array x not equally spaced")
    if ((np.min(xnew) < np.min(x)) | (np.max(xnew) > np.max(x))):
        raise Exception("xnew out of bounds")
        
    # interval and data position
    h = np.diff(x)[0]
    L = x.size
    s = (xnew - np.min(x)) / h
    
    # weight matrix
    si = np.abs(np.resize(s, [L + 2, np.size(s)]).transpose() 
                - np.resize(np.linspace(-1, L, L + 2), [np.size(s), L + 2]))
    pos = si <= 1
    si[pos] = 1.5*np.power(si[pos], 3) - 2.5*np.power(si[pos], 2) + 1
    pos = (si > 1) & (si <= 2)
    si[pos] = - 0.5*np.power(si[pos], 3) + 2.5*np.power(si[pos], 2) - 4*si[pos] + 2
    si[si > 2] = 0
    #W = scipy.sparse.coo_matrix(si)
    W = si
    
    # borders
    W[:, 1] = W[:, 1] + 3*W[:, 0]
    W[:, 2] = W[:, 2] - 3*W[:, 0]
    W[:, 3] = W[:, 3] + 1*W[:, 0]
    W[:, L] = W[:, L] + 3*W[:, L + 1]
    W[:, L - 1] = W[:, L - 1] - 3*W[:, L + 1]
    W[:, L - 2] = W[:, L - 2] + 1*W[:, L + 1]
    W = W[:, range(1, L + 1)]
    
    return W