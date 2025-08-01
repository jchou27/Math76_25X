## Python translations of some MATLAB functions contained in Per Christian Hansen's regtools package.
## These have not been thoroughly tested to reproduce the MATLAB (only tested this on a few examples).
## Use at your own risk!
## Jonathan Lindbloom, 10/16/24

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.optimize import fminbound
from scipy.interpolate import splrep, splev, splder
from scipy.linalg import qr
from scipy.linalg import norm
import warnings



def extract_scalar(arr):
    if isinstance(arr, np.ndarray) and arr.size == 1:
        return arr.item()  # Extract the scalar from a 1x1 array
    elif isinstance(arr, list) and len(arr) == 1:
        return arr[0] # Extract the scalar from a list
    return arr  # Return the input directly if it is already a scalar



def csvd(A, tst=None):
    """
    Computes the compact form of the Singular Value Decomposition (SVD) of matrix A:
        A = U @ np.diag(s) @ V.T
    where U, s, V are the SVD components with the following dimensions:
        U  is m-by-min(m, n)
        s  is min(m, n)-by-1
        V  is n-by-min(m, n).
    
    If a second argument is provided, the full U and V matrices are returned.
    
    Parameters:
    A : array_like
        Input matrix of shape (m, n).
    tst : optional
        If provided, returns the full SVD matrices.
    
    Returns:
    U : ndarray
        Left singular vectors.
    s : ndarray
        Singular values.
    V : ndarray
        Right singular vectors.
    """
    m, n = A.shape
    if tst is None:
        # Compact SVD
        if m >= n:
            U, s, Vh = svd(A, full_matrices=False)
        else:
            Vh, s, U = svd(A.T, full_matrices=False)
            Vh = Vh.T
    else:
        # Full SVD
        U, s, Vh = svd(A, full_matrices=True)

    return U, s, Vh.T






def deriv2(n, example=1):
    """
    Test problem: computation of the second derivative.

    Parameters:
    n (int): Size of the matrix and vectors.
    example (int): Specifies the right-hand side vector and solution vector.

    Returns:
    A (ndarray): The n x n matrix.
    b (ndarray): The right-hand side vector.
    x (ndarray): The solution vector.
    """
    # Initialization
    h = 1.0 / n
    sqh = np.sqrt(h)
    h32 = h * sqh
    h2 = h ** 2
    sqhi = 1.0 / sqh
    t = 2.0 / 3.0
    A = np.zeros((n, n))

    # Compute the matrix A
    for i in range(n):
        A[i, i] = h2 * ( ( (i + 1)**2 - (i + 1) + 0.25) * h - (i + 1 - t) )
        for j in range(i):
            A[i, j] = h2 * (j + 0.5) * ((i + 0.5) * h - 1)
    A = A + np.tril(A, -1).T

    # Compute the right-hand side vector b
    b = np.zeros(n)
    if example == 1:
        for i in range(n):
            b[i] = h32 * (i + 0.5) * (((i + 1)**2 + i**2) * h2 / 2 - 1) / 6
    elif example == 2:
        ee = 1 - np.exp(1)
        for i in range(n):
            b[i] = sqhi * (np.exp((i + 1) * h) - np.exp(i * h) + ee * (i + 0.5) * h2 - h)
    elif example == 3:
        if n % 2 != 0:
            raise ValueError("Order n must be even")
        else:
            for i in range(n // 2):
                s12 = ((i + 1) * h) ** 2
                s22 = (i * h) ** 2
                b[i] = sqhi * (s12 + s22 - 1.5) * (s12 - s22) / 24
            for i in range(n // 2, n):
                s1 = (i + 1) * h
                s12 = s1 ** 2
                s2 = i * h
                s22 = s2 ** 2
                b[i] = sqhi * (-(s12 + s22) * (s12 - s22) + 4 * (s1**3 - s2**3) - 4.5 * (s12 - s22) + h) / 24
    else:
        raise ValueError("Illegal value of example")

    # Compute the solution vector x
    x = np.zeros(n)
    if example == 1:
        for i in range(n):
            x[i] = h32 * (i + 0.5)
    elif example == 2:
        for i in range(n):
            x[i] = sqhi * (np.exp((i + 1) * h) - np.exp(i * h))
    elif example == 3:
        for i in range(n // 2):
            x[i] = sqhi * (((i + 1) * h) ** 2 - (i * h) ** 2) / 2
        for i in range(n // 2, n):
            x[i] = sqhi * (h - (((i + 1) * h) ** 2 - (i * h) ** 2) / 2)

    return A, b, x





def fil_fac(s, reg_param, method='Tikh', s1=None, V1=None):
    """
    Computes the filter factors for some regularization methods.

    Parameters:
    s (ndarray): Singular values or a matrix containing singular values.
    reg_param (ndarray): Regularization parameters.
    method (str): Regularization method. Default is 'Tikh'.
    s1 (ndarray, optional): Singular values for 'ttls' method.
    V1 (ndarray, optional): Right singular matrix for 'ttls' method.

    Returns:
    f (ndarray): The filter factors matrix.
    """
    if np.isscalar(reg_param):
        reg_param = [reg_param]

    p, ps = s.shape if s.ndim > 1 else (len(s), 1)
    lr = len(reg_param)
    f = np.zeros((p, lr))

    # Check input data
    if np.min(reg_param) <= 0:
        raise ValueError("Regularization parameter must be positive")
    if (method.startswith(('tsvd', 'tgsv', 'ttls')) and np.max(reg_param) > p):
        raise ValueError("Truncation parameter too large")

    # Compute the filter factors
    for j in range(lr):
        if method.startswith(('cg', 'nu', 'ls')):
            raise ValueError("Filter factors for iterative methods are not supported")
        elif method.startswith(('dsvd', 'dgsv')):
            if ps == 1:
                f[:, j] = s / (s + reg_param[j])
            else:
                f[:, j] = s[:, 0] / (s[:, 0] + reg_param[j] * s[:, 1])
        elif method.lower().startswith('tikh'):
            if ps == 1:
                f[:, j] = (s ** 2) / (s ** 2 + reg_param[j] ** 2)
            else:
                f[:, j] = (s[:, 0] ** 2) / (s[:, 0] ** 2 + reg_param[j] ** 2 * s[:, 1] ** 2)
        elif method.startswith(('tsvd', 'tgsv')):
            if ps == 1:
                f[:reg_param[j], j] = 1
                f[reg_param[j]:, j] = 0
            else:
                f[:p - reg_param[j], j] = 0
                f[p - reg_param[j]:, j] = 1
        elif method.startswith('ttls'):
            if s1 is not None and V1 is not None:
                coef = (V1[p, :] ** 2) / np.linalg.norm(V1[p, reg_param[j]:p + 1]) ** 2
                for i in range(p):
                    k = reg_param[j]
                    f[i, j] = s[i] ** 2 * np.sum(coef[:k] / (s1[:k] + s[i]) / (s1[:k] - s[i]))
                    if f[i, j] < 0:
                        f[i, j] = np.finfo(float).eps
                    if i > 0:
                        if f[i - 1, j] <= np.finfo(float).eps and f[i, j] > f[i - 1, j]:
                            f[i, j] = f[i - 1, j]
            else:
                raise ValueError("The SVD of [A, b] must be supplied")
        elif method.startswith('mtsv'):
            raise ValueError("Filter factors for MTSVD are not supported")
        else:
            raise ValueError("Illegal method")

    return f






def l_corner(rho, eta, reg_param=None, U=None, s=None, b=None, method='Tikh', M=None):
    """
    Locate the "corner" of the L-curve in log-log scale.

    Parameters:
    rho : array_like
        Corresponding values of || A x - b ||.
    eta : array_like
        Corresponding values of || L x ||.
    reg_param : array_like, optional
        The regularization parameters.
    U : array_like, optional
        The left singular vectors.
    s : array_like, optional
        The singular values.
    b : array_like, optional
        The right-hand side vector.
    method : str, optional
        Regularization method (default is 'Tikh').
    M : float, optional
        Upper bound for eta, below which the corner should be found.

    Returns:
    reg_c : float
        The regularization parameter corresponding to the corner.
    rho_c : float
        The value of || A x - b || at the corner.
    eta_c : float
        The value of || L x || at the corner.
    """

    # Ensure that rho and eta are column vectors.
    rho = np.array(rho).flatten()
    eta = np.array(eta).flatten()

    # Set default regularization parameter if needed.
    if reg_param is None:
        reg_param = np.arange(1, len(rho) + 1)

    if method is None:
        method = 'Tikh'

    # Set threshold for small singular values
    s_thr = np.finfo(float).eps

    # Restrict the analysis of the L-curve according to M (if specified).
    if M is not None:
        indices = np.where(eta < M)[0]
        rho = rho[indices]
        eta = eta[indices]
        reg_param = reg_param[indices]

    if method.lower().startswith('tikh'):

        beta = U.T @ b
        b0 = b - U @ beta
        xi = beta / s

        # Compute curvature of the L-curve.
        g = [lcfun(rp, s, beta, xi) for rp in reg_param]

        # fig2, axs2 = plt.subplots()
        # axs2.plot(reg_param, np.asarray(g))
        # fig2.show()

        # Locate the corner
        gi_min = np.argmin(g)
        reg_c = fminbound(lcfun, reg_param[min(gi_min+1, len(g))], reg_param[max(gi_min - 1, 1)],
                          args=(s, beta, xi), disp=False)
        
        kappa_max = -lcfun(reg_c, s, beta, xi)

        if kappa_max < 0:
            lr = len(rho)
            reg_c = reg_param[lr - 1]
            rho_c = rho[lr - 1]
            eta_c = eta[lr - 1]
        else:
            f = s**2 / (s**2 + reg_c**2)
            eta_c = np.linalg.norm(f * xi)
            rho_c = np.linalg.norm((1 - f) * beta[:len(f)])
            if U.shape[0] > U.shape[1]:
                rho_c = np.sqrt(rho_c**2 + np.linalg.norm(b0)**2)

    elif method.lower().startswith('tsvd') or method.lower().startswith('mtsv'):
        
        # Use spline fitting if no splines toolbox available, fallback to pruning algorithm.

        if len(rho) < 4:
            raise ValueError("Too few data points for L-curve analysis")

        # Convert to logarithms.
        lrho = np.log(rho)
        leta = np.log(eta)

        # Fit a spline to the log-log curve.
        tck = splrep(lrho, leta)
        d_lrho = splev(lrho, splder(tck, 1))  # First derivative
        dd_lrho = splev(lrho, splder(tck, 2))  # Second derivative

        # Compute curvature
        kappa = - (d_lrho * dd_lrho) / ((d_lrho**2 + dd_lrho**2)**(1.5))
        kappa[np.isnan(kappa)] = 0  # Set invalid values to 0

        # Find maximum curvature
        ikmax = np.argmax(kappa)
        reg_c = reg_param[ikmax]
        rho_c = rho[ikmax]
        eta_c = eta[ikmax]

    elif method.lower().startswith('dsvd'):
        # Same as for Tikhonov, but with different treatment of singular values.
        beta = U.T @ b
        b0 = b - U @ beta
        xi = beta / s

        # Get curvature
        g = [lcfun(rp, s, beta, xi) for rp in reg_param]

        gi_min = np.argmin(g)
        reg_c = fminbound(lcfun, reg_param[max(gi_min - 1, 0)], reg_param[min(gi_min + 1, len(g) - 1)],
                          args=(s, beta, xi), disp=False)
        kappa_max = -lcfun(reg_c, s, beta, xi)

        if kappa_max < 0:
            lr = len(rho)
            reg_c = reg_param[lr - 1]
            rho_c = rho[lr - 1]
            eta_c = eta[lr - 1]
        else:
            f = s / (s + reg_c)
            eta_c = np.linalg.norm(f * xi)
            rho_c = np.linalg.norm((1 - f) * beta[:len(f)])
            if U.shape[0] > U.shape[1]:
                rho_c = np.sqrt(rho_c**2 + np.linalg.norm(b0)**2)
    else:
        raise ValueError("Illegal method")

    return reg_c, rho_c, eta_c







def lcfun(lambd, s, beta, xi, fifth=False):
    """
    Auxiliary routine for l_corner; computes the NEGATIVE of the curvature.
    Parameters:
    lambd : float or array_like
        Regularization parameter(s).
    s : array_like
        Singular values.
    beta : array_like
        Coefficients from U.T @ b.
    xi : array_like
        Coefficients from beta / s.
    fifth : bool, optional
        If True, uses an alternative formula for computing f.
    
    Returns:
    g : array_like
        The negative of the curvature.
    """
    
    lambd = np.array(lambd)
    lambd = np.atleast_1d(lambd)
    phi = np.zeros_like(lambd)
    dphi = np.zeros_like(lambd)
    psi = np.zeros_like(lambd)
    dpsi = np.zeros_like(lambd)
    eta = np.zeros_like(lambd)
    rho = np.zeros_like(lambd)
    
    # Handle possible least squares residual.
    if len(beta) > len(s):  # A possible least squares residual.
        LS = True
        rhoLS2 = beta[-1]**2
        beta = beta[:-1]
    else:
        LS = False
    
    # Compute intermediate quantities.
    for i in range(len(lambd)):
        if not fifth:
            f = (s**2) / (s**2 + lambd[i]**2)
        else:
            f = s / (s + lambd[i])
        
        cf = 1 - f
        eta[i] = np.linalg.norm(f * xi)
        rho[i] = np.linalg.norm(cf * beta)
        
        f1 = -2 * f * cf / lambd[i]
        f2 = -f1 * (3 - 4 * f) / lambd[i]
        
        phi[i] = np.sum(f * f1 * np.abs(xi)**2)
        psi[i] = np.sum(cf * f1 * np.abs(beta)**2)
        dphi[i] = np.sum((f1**2 + f * f2) * np.abs(xi)**2)
        dpsi[i] = np.sum((-f1**2 + cf * f2) * np.abs(beta)**2)
    
    if LS:  # Include least squares residual if applicable.
        rho = np.sqrt(rho**2 + rhoLS2)
    
    # Compute first and second derivatives of eta and rho w.r.t. lambda.
    deta = phi / eta
    drho = -psi / rho
    ddeta = (dphi / eta) - (deta**2 / eta)
    ddrho = (-dpsi / rho) - (drho**2 / rho)
    
    # Convert to derivatives of log(eta) and log(rho).
    dlogeta = deta / eta
    dlogrho = drho / rho
    ddlogeta = (ddeta / eta) - (dlogeta**2)
    ddlogrho = (ddrho / rho) - (dlogrho**2)
    
    # Compute curvature (negative).
    numerator = dlogrho * ddlogeta - ddlogrho * dlogeta
    denominator = (dlogrho**2 + dlogeta**2)**1.5
    g = -numerator / denominator
    
    return g








def l_curve(U, sm, b, method='Tikh', L=None, V=None, fig=None):
    """
    Plot the L-curve and find its "corner".
    
    Parameters:
    U : array_like
        Matrix of left singular vectors (or generalized case).
    sm : array_like
        Singular values or generalized singular values (depending on the method).
    b : array_like
        Right-hand side vector.
    method : str, optional
        Regularization method ('Tikh', 'tsvd', 'dsvd', 'mtsvd'). Default is 'Tikh'.
    L : array_like, optional
        Matrix for modified TSVD (only for 'mtsvd' method).
    V : array_like, optional
        Matrix of right singular vectors (only for 'mtsvd' method).
    
    Returns:
    reg_corner : float
        Regularization parameter corresponding to the corner of the L-curve.
    rho : array_like
        Residual norms || A x - b ||.
    eta : array_like
        Solution norms || L x ||.
    reg_param : array_like
        Regularization parameters.
    """
    if fig is None:
        fig, axs = plt.subplots()

    npoints = 200  # Number of points on the L-curve for 'Tikh' and 'dsvd'.
    smin_ratio = 16 * np.finfo(float).eps  # Smallest regularization parameter.

    # Initialization
    m, n = U.shape
    p, ps = sm.shape if sm.ndim > 1 else (len(sm), 1)
    beta = U.T @ b
    beta2 = norm(b)**2 - norm(beta)**2

    if ps == 1:
        s = sm
        beta = beta[:p]
    else:
        s = sm[::-1, 0] / sm[::-1, 1]
        beta = beta[::-1]

    xi = beta[:p] / s
    xi[np.isinf(xi)] = 0  # Handle infinite values by setting to 0

    # Handle different methods
    if method.lower().startswith('tikh'):
        eta = np.zeros(npoints)
        rho = np.zeros(npoints)
        reg_param = np.zeros(npoints)
        s2 = s**2
        reg_param[-1] = max(s[-1], s[0] * smin_ratio)
        ratio = (s[0] / reg_param[-1])**(1 / (npoints - 1))

        for i in range(npoints - 2, -1, -1):
            reg_param[i] = ratio * reg_param[i + 1]

        for i in range(npoints):
            f = s2 / (s2 + reg_param[i]**2)
            eta[i] = norm(f * xi)
            rho[i] = norm((1 - f) * beta[:p])

        if m > n and beta2 > 0:
            rho = np.sqrt(rho**2 + beta2)

        marker = '-'
        txt = 'Tikh.'

    elif method.lower().startswith('tsvd'):
        eta = np.zeros(p)
        rho = np.zeros(p)
        eta[0] = np.abs(xi[0])**2

        for k in range(1, p):
            eta[k] = eta[k - 1] + np.abs(xi[k])**2
        eta = np.sqrt(eta)

        if m > n:
            rho[p - 1] = beta2 if beta2 > 0 else np.finfo(float).eps**2
        else:
            rho[p - 1] = np.finfo(float).eps**2

        for k in range(p - 2, -1, -1):
            rho[k] = rho[k + 1] + np.abs(beta[k + 1])**2

        rho = np.sqrt(rho)
        reg_param = np.arange(1, p + 1)
        marker = 'o'
        txt = 'TSVD' if ps == 1 else 'TGSVD'

    elif method.lower().startswith('dsvd'):
        eta = np.zeros(npoints)
        rho = np.zeros(npoints)
        reg_param = np.zeros(npoints)
        reg_param[-1] = max(s[-1], s[0] * smin_ratio)
        ratio = (s[0] / reg_param[-1])**(1 / (npoints - 1))

        for i in range(npoints - 2, -1, -1):
            reg_param[i] = ratio * reg_param[i + 1]

        for i in range(npoints):
            f = s / (s + reg_param[i])
            eta[i] = norm(f * xi)
            rho[i] = norm((1 - f) * beta[:p])

        if m > n and beta2 > 0:
            rho = np.sqrt(rho**2 + beta2)

        marker = ':'
        txt = 'DSVD' if ps == 1 else 'DGSVD'

    elif method.lower().startswith('mtsv'):
        if L is None or V is None:
            raise ValueError("The matrices L and V must also be specified for 'mtsvd' method.")
        
        rho = np.zeros(p)
        eta = np.zeros(p)
        Q, R = qr(L @ V[:, n - p:], mode='economic')

        for i in range(p):
            k = n - p + i
            Lxk = L @ V[:, :k] @ xi[:k]
            zk = np.linalg.solve(R[:n - k, :n - k], Q[:, :n - k].T @ Lxk)
            zk = zk[::-1]

            eta[i] = norm(Q[:, n - k:].T @ Lxk)
            if i < p - 1:
                rho[i] = norm(beta[k + 1:n] + s[k + 1:n] * zk)
            else:
                rho[i] = np.finfo(float).eps

        if m > n and beta2 > 0:
            rho = np.sqrt(rho**2 + beta2)

        reg_param = np.arange(n - p + 1, n + 1)
        txt = 'MTSVD'
        marker = 'x'
        ps = 2  # General form regularization.

    else:
        raise ValueError('Illegal method')

    # Locate the "corner" of the L-curve (if requested).
    reg_corner, rho_c, eta_c = None, None, None
    if npoints > 0:
        reg_corner, rho_c, eta_c = l_corner(rho, eta, reg_param, U, sm, b, method)

    # Plot L-curve
    plot_lc(rho, eta, marker, ps, reg_param, ax=axs)
    if reg_corner:
        axs.loglog([min(rho) / 100, rho_c], [eta_c, eta_c], ':r',
                   [rho_c, rho_c], [min(eta) / 100, eta_c], ':r')
        axs.set_title(f'L-curve, {txt} corner at $\lambda \\approx$ {extract_scalar(reg_corner):.3e}')

    return reg_corner, rho, eta, reg_param



def plot_lc(rho, eta, marker, ps, reg_param, ax=None):
    """
    Helper function to plot the L-curve.
    """
    if ax is None:
        plt.loglog(rho, eta, marker)
        plt.xlabel('Residual Norm || A x - b ||')
        plt.ylabel('Solution Norm || x ||')
        plt.show()
    else:
        ax.loglog(rho, eta, marker)
        ax.set_xlabel('Residual Norm || A x - b ||')
        ax.set_ylabel('Solution Norm || x ||')








def tikhonov(U, s, V, b, lambdas, x_0=None):
    """
    Tikhonov regularization.

    Parameters:
    U : array_like
        Left singular vectors (from SVD or GSVD).
    s : array_like
        Singular values (or generalized singular values).
    V : array_like
        Right singular vectors (from SVD or GSVD).
    b : array_like
        Right-hand side vector.
    lambdas : array_like
        Regularization parameter (or vector of parameters).
    x_0 : array_like, optional
        Initial estimate for x. If not provided, assumed to be zero.

    Returns:
    x_lambda : array_like
        Tikhonov regularized solution for each lambda.
    rho : array_like
        Residual norms for each lambda.
    eta : array_like
        Solution norms (or seminorms for general-form regularization).
    """
    if np.isscalar(lambdas):
        lambdas = np.asarray([lambdas])

    # Ensure lambda is non-negative
    if np.any(lambdas < 0):
        raise ValueError("Illegal regularization parameter lambda")

    # Initialization
    m, n = U.shape[0], V.shape[0]
    p, ps = s.shape if len(s.shape) == 2 else (len(s), 1)
    beta = U[:, :p].T @ b
    zeta = s[:, 0] * beta if ps > 1 else s * beta
    ll = len(lambdas)
    x_lambda = np.zeros((n, ll))
    rho = np.zeros(ll)
    eta = np.zeros(ll)

    # Handle the standard-form case
    if ps == 1:
        omega = V.T @ x_0 if x_0 is not None else None
        for i in range(ll):
            if x_0 is None:
                x_lambda[:, i] = V[:, :p] @ (zeta / (s**2 + lambdas[i]**2))
                rho[i] = lambdas[i]**2 * np.linalg.norm(beta / (s**2 + lambdas[i]**2))
            else:
                x_lambda[:, i] = V[:, :p] @ ((zeta + lambdas[i]**2 * omega) / (s**2 + lambdas[i]**2))
                rho[i] = lambdas[i]**2 * np.linalg.norm((beta - s * omega) / (s**2 + lambdas[i]**2))
            eta[i] = np.linalg.norm(x_lambda[:, i])

        if len(U) > p:
            rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:, :p] @ beta)**2)

    elif m >= n:
        # The overdetermined or square general-form case
        gamma2 = (s[:, 0] / s[:, 1])**2
        omega = np.linalg.solve(V, x_0)[:p] if x_0 is not None else None
        x0 = np.zeros(n) if p == n else V[:, p:] @ (U[:, p:].T @ b)

        for i in range(ll):
            if x_0 is None:
                xi = zeta / (s[:, 0]**2 + lambdas[i]**2 * s[:, 1]**2)
                x_lambda[:, i] = V[:, :p] @ xi + x0
                rho[i] = lambdas[i]**2 * np.linalg.norm(beta / (gamma2 + lambdas[i]**2))
            else:
                xi = (zeta + lambdas[i]**2 * s[:, 1]**2 * omega) / (s[:, 0]**2 + lambdas[i]**2 * s[:, 1]**2)
                x_lambda[:, i] = V[:, :p] @ xi + x0
                rho[i] = lambdas[i]**2 * np.linalg.norm((beta - s[:, 0] * omega) / (gamma2 + lambdas[i]**2))
            eta[i] = np.linalg.norm(s[:, 1] * xi)

        if len(U) > p:
            rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:, :p] @ beta)**2)

    else:
        # The underdetermined general-form case
        gamma2 = (s[:, 0] / s[:, 1])**2
        if x_0 is not None:
            raise ValueError("x_0 not allowed in underdetermined case")

        x0 = np.zeros(n) if p == m else V[:, p:] @ (U[:, p:].T @ b)

        for i in range(ll):
            xi = zeta / (s[:, 0]**2 + lambdas[i]**2 * s[:, 1]**2)
            x_lambda[:, i] = V[:, :p] @ xi + x0
            rho[i] = lambdas[i]**2 * np.linalg.norm(beta / (gamma2 + lambdas[i]**2))
            eta[i] = np.linalg.norm(s[:, 1] * xi)

    return x_lambda, rho, eta




def wing(n, t1=None, t2=None):
    """
    Test problem with a discontinuous solution.

    Parameters:
    n : int
        Number of discretization points.
    t1 : float, optional
        Left boundary for the discontinuity in the solution. Default is 1/3.
    t2 : float, optional
        Right boundary for the discontinuity in the solution. Default is 2/3.

    Returns:
    A : array_like
        Discretized operator matrix.
    b : array_like
        Right-hand side vector (optional, only if requested).
    x : array_like
        Discontinuous solution (optional, only if requested).
    """
    # Set default values for t1 and t2 if not provided
    if t1 is None and t2 is None:
        t1 = 1/3
        t2 = 2/3
    elif t1 is not None and t2 is not None and t1 > t2:
        raise ValueError("t1 must be smaller than t2")

    # Initialize A and set up step size h
    A = np.zeros((n, n))
    h = 1.0 / n

    # Set up matrix
    sti = (np.arange(1, n + 1) - 0.5) * h  # Midpoint rule for s and t
    for i in range(n):
        A[i, :] = h * sti * np.exp(-sti[i] * sti**2)

    # Set up right-hand side (b)
    b = None
    if t1 is not None and t2 is not None:
        b = np.sqrt(h) * 0.5 * (np.exp(-sti * t1**2) - np.exp(-sti * t2**2)) / sti

    # Set up solution (x)
    x = None
    if t1 is not None and t2 is not None:
        I = np.where((t1 < sti) & (sti < t2))[0]
        x = np.zeros(n)
        x[I] = np.sqrt(h) * np.ones(len(I))

    return A, b, x






def tsvd(U, s, V, b, k):
    """
    Truncated SVD regularization.

    Parameters:
    U : array_like
        Left singular vectors (from SVD).
    s : array_like
        Singular values.
    V : array_like
        Right singular vectors (from SVD).
    b : array_like
        Right-hand side vector.
    k : array_like
        Truncation parameter(s). Can be a scalar or a list/array of integers.

    Returns:
    x_k : array_like
        Truncated SVD solution for each k.
    rho : array_like
        Residual norms for each k.
    eta : array_like
        Solution norms for each k.
    """
    n, p = V.shape
    lk = len(k) if isinstance(k, (list, np.ndarray)) else 1  # Handle scalar k
    k = np.atleast_1d(k)  # Ensure k is treated as a vector

    # Ensure truncation parameter k is valid
    if np.min(k) < 0 or np.max(k) > p:
        raise ValueError("Illegal truncation parameter k")

    # Initialize outputs
    x_k = np.zeros((n, lk))
    eta = np.zeros(lk)
    rho = np.zeros(lk)

    # Compute initial values
    beta = U[:, :p].T @ b
    xi = beta / s  # Scaling by the singular values

    # Loop through each value of k
    for j in range(lk):
        i = k[j]
        if i > 0:
            # Truncated SVD solution
            x_k[:, j] = V[:, :i] @ xi[:i]
            eta[j] = np.linalg.norm(xi[:i])
            rho[j] = np.linalg.norm(beta[i:p])

    # Compute residual norms if necessary
    if U.shape[0] > p:
        rho = np.sqrt(rho**2 + np.linalg.norm(b - U[:, :p] @ beta)**2)

    return x_k, rho, eta






