## Python translations of some MATLAB functions contained in Per Christian Hansen's regtools package.
## These have not been thoroughly tested to reproduce the MATLAB (only tested this on a few examples).
## Use at your own risk!
## Jonathan Lindbloom, 10/16/24


import numpy as np
import matplotlib.pyplot as plt


def build_gravity_matrix(n, d):
    """Returns the matrix for the gravity problem, with parameters n and d.
    """

    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i,j] = (d/n)*np.power( (d**2) + ((( i+1 - (j+1) )/n)**2) , -1.5 )

    return A


def shaw(n):
    """
    Test problem: one-dimensional image restoration model.

    Discretization of a first-kind Fredholm integral equation with
    [-pi/2, pi/2] as both integration intervals.

    Parameters:
    n -- Order of the matrix (must be even).

    Returns:
    A -- Discretized matrix from the Fredholm integral equation.
    b -- Right-hand side vector.
    x -- Solution vector.

    Reference: C. B. Shaw, Jr., "Improvements of the resolution of
    an instrument by numerical solution of an integral equation",
    J. Math. Anal. Appl. 37 (1972), 83-112.

    Adapted from shaw.m in regtools, by Per Christian Hansen.
    """
    
    assert n%2 == 0, "input n must be even!"
    nhalf = int(n/2.0)

    # Initialization.
    h = np.pi/n
    A = np.zeros((n,n))

    # Compute the matrix A.
    co = np.cos(-(np.pi/2) + (np.arange(0.5, n-0.5+1, 1)*h)  )
    psi = np.pi*np.sin(-(np.pi/2) + (np.arange(0.5, n-0.5+1, 1)*h))

    for i in range(nhalf):
        for j in range(i,(n-1)-i):
            ss = psi[i] + psi[j]
            A[i,j] = ((co[i] + co[j])*np.sin(ss)/ss)**2
            A[(n-1)-j,(n-1)-i] = A[i,j]
         
        A[i, (n-1)-i] = (2*co[i])**2

    A = A.copy() + np.triu(A,1).T
    A = A*h

    # Compute the vectors x and b
    a1 = 2 
    c1 = 6
    t1 =  .8
    a2 = 1
    c2 = 2
    t2 = -.5

    x = ( a1*np.exp(-c1*(-np.pi/2 + (np.arange(0.5, n-0.5+1, 1)*h) - t1)**2) 
        + a2*np.exp(-c2*(-np.pi/2 + (np.arange(0.5, n-0.5+1, 1)*h) - t2)**2) )
                      
    b = A @ x

    return A, b, x



def picard(U, s, b, d=0):
    """
    Visual inspection of the Picard condition.
    
    Plots the singular values, the absolute value of the Fourier coefficients,
    and a (possibly smoothed) curve of the solution coefficients eta.
    
    Parameters:
    U -- Left singular vectors (from SVD or GSVD)
    s -- Singular values (or generalized singular values [sigma, mu])
    b -- Right-hand side vector
    d -- Smoothing parameter (default is 0, no smoothing)
    
    Returns:
    eta -- Solution coefficients eta(i) = |U[:, i].T * b| / s(i)

    Adapted from picard.m in regtools, by Per Christian Hansen.
    """
    
    # Initialization
    n, ps = s.shape if len(s.shape) > 1 else (len(s), 1)
    beta = np.abs(U[:, :n].T @ b)
    eta = np.zeros(n)
    
    # If s has two columns, compute generalized singular values
    if ps == 2:
        s = s[:, 0] / s[:, 1]
    
    d21 = 2 * d + 1
    keta = np.arange(1 + d, n - d)
    
    # Avoid division by zero warnings
    if not np.all(s):
        print("Warning: Division by zero singular values.")
    
    # Compute the eta values with geometric mean smoothing
    for i in keta:
        eta[i] = (np.prod(beta[i - d:i + d + 1]) ** (1 / d21)) / s[i]
    
    # Plot the data using a semilogarithmic scale
    plt.semilogy(np.arange(1, n + 1), s, '.-', label=r'$\sigma_i$')
    plt.semilogy(np.arange(1, n + 1), beta, 'x', label=r'$|u_i^Tb|$')
    plt.semilogy(keta + 1, eta[keta], 'o', fillstyle='none', label=r'$|u_i^Tb|/\sigma_i$')
    
    plt.xlabel('i')
    plt.title('Picard plot')
    
    # Add the legend
    if ps == 1:
        plt.legend([r'$\sigma_i$', r'$|u_i^Tb|$', r'$|u_i^Tb|/\sigma_i$'])
    else:
        plt.legend([r'$\sigma_i/\mu_i$', r'$|u_i^Tb|$', r'$|u_i^Tb| / (\sigma_i/\mu_i)$'], loc='northwest')
    
    plt.show()
    
    return eta




def count_sign_switches(vec):
    # Convert the vector to a NumPy array if it's not already one
    vec = np.asarray(vec)
    
    # Compute the product of consecutive elements
    product = vec[:-1] * vec[1:]
    
    # A sign switch happens when the product of consecutive elements is negative
    sign_switches = np.sum(product < 0)
    
    return sign_switches


