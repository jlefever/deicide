"""Grand Index (13GRI) and Adjusted Grand Index (13AGRI).

Author : Victor Alexandre Padilha <victorpadilha.cc@gmail.com>
License: BSD 3-Clause
"""

import numpy as np

__version__ = '1.0'

def grand_index(u, v, adjusted=False):
    """
    Grand Index (13GRI) and Adjusted Grand Index (13AGRI) implementations.
    
    These measures are capable of comparing exclusive hard, fuzzy/probabilistic,
    non-exclusive hard, and possibilistic clustering solutions.
    
    Variable names used in this implementation directly follow the notations used
    in the original paper of the indices. For more information, you should check
    the reference.
    
    Reference
    ---------
    Horta, D., and Campello, R. J. G. B. (2015). Comparing Hard and Overlapping Clusterings.
    Journal of Machine Learning Research, 16: 2949-2997.
    
    Parameters
    ----------
    u : numpy.ndarray
        k x n array representing a clustering solution with k clusters for a dataset
        with n objects. u[r, i] expresses the membership degree of the ith object to
        the rth cluster.
    
    v : numpy.ndarray
        l x n array representing a clustering solution with l clusters for a dataset
        with n objects. v[r, i] expresses the membership degree of the ith object to
        the rth cluster.
    
    adjusted : bool, default: False
        If True, calculates 13AGRI. Otherwise, calculates 13GRI.
    
    Returns
    -------
    gri : float
        Similarity score of U and V. A score of 1 represents a perfect match.
        If adjusted is True, random solutions are expected to achieve a score close to 0.
    """
    u = u.astype(np.double, copy=True)
    v = v.astype(np.double, copy=True)    
    
    _validate_parameters(u, v)
    
    ju, su, tu = _calculate_information_arrays(u)
    jv, sv, tv = _calculate_information_arrays(v)
    tmax = max(np.sum(tu), np.sum(tv))
    gri = _calculate_gri(ju, su, jv, sv, tmax)

    if not adjusted:
        return gri

    gri_expectation = _calculate_gri_expectation(ju, su, jv, sv, tmax)
    return (gri - gri_expectation) / (1.0 - gri_expectation)

def _calculate_information_arrays(u):
    k, n = u.shape
    i = np.triu_indices(n, 1)
    ju = np.dot(u.T, u)[i]
    su = np.dot(u.T, np.dot(np.ones((k, k)) - np.identity(k), u))[i]
    return ju, su, ju + su

def _calculate_gri(ju, su, jv, sv, tmax):
    a = np.sum(np.minimum(ju, jv))
    d = np.sum(np.minimum(su, sv))
    return (a + d) / tmax

def _calculate_gri_expectation(ju, su, jv, sv, tmax):
    a_expectation = _calculate_expectation(ju, jv)
    d_expectation = _calculate_expectation(su, sv)
    return (a_expectation + d_expectation) / tmax

def _calculate_expectation(ju, jv):
    x, y = np.sort(ju), np.sort(jv)
    m = len(ju)

    expectation = 0.0

    j = m - 1
    for i in reversed(range(m)):
        while j >= 0 and x[i] <= y[j]:
            j -= 1        
        expectation += (m - j - 1) * x[i]
    
    i = m - 1
    for j in reversed(range(m)):
        while i >= 0 and x[i] > y[j]:
            i -= 1
        expectation += (m - i - 1) * y[j]
    
    return expectation / m

def _validate_parameters(u, v):
    if u.shape[1] != v.shape[1]:
        raise ValueError("arrays 'u' and 'v' must have the same number of columns")
    
    if np.any(u < 0.0) or np.any(u > 1.0) or np.any(v < 0.0) or np.any(v > 1.0):
        raise ValueError("all elements from 'u' and 'v' must assume values between 0.0 or 1.0")
