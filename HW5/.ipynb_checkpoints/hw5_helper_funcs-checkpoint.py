import numpy as np
import numpy.linalg as npl
import scipy.linalg as spla
import scipy.fft as sfft
import matplotlib.pyplot as plt

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception as e:
    _HAS_CVXPY = False
    print("Warning: cvxpy not available. Install it with `pip install cvxpy` to run the ℓ1/TV and complex LS problems.")

# Relative Error
def relerr(xhat, x):
    return np.linalg.norm(xhat - x)/np.linalg.norm(x)

# Signal builders
def signal_sparse(n=200, s=10, scale=1.0):
    idx = np.random.permutation(n)[:s]
    f = np.zeros(n)
    f[idx] = scale * np.random.randn(s)
    return f

def signal_piecewise_constant(n=200):
    # f(t)=1 on (-1/4,0], f(t)=2 on [1/2,7/8], else 0; periodic in [-1,1]
    t = -1 + 2*np.arange(n)/n
    f = np.zeros_like(t)
    f[(t > -0.25) & (t <= 0.0)] = 1.0
    f[(t >= 0.5) & (t <= 0.875)] = 2.0
    return f

def signal_sawtooth(n=200):
    # 2-periodic sawtooth on (-1,1]: f(t) = -1-t for -1<t<=0; f(t)=1-t for 0<t<=1
    t = -1 + 2*np.arange(n)/n
    f = np.where(t<=0, -1 - t, 1 - t)
    return f




# Solvers 
def solve_ls(A, g):
    # Least-squares (stable)
    return npl.lstsq(A, g, rcond=None)[0]

def solve_tikhonov(A, L, g, lam):
    # (A^T A + lam^2 L^T L) f = A^T g
    ATA = A.T @ A
    LTL = L.T @ L
    rhs = A.T @ g
    return spla.solve(ATA + (lam**2)*LTL, rhs, assume_a='pos')

def tikh_complex_solve(F, L, b, lam):
    # Solve (F^H F + lam^2 L^T L) f = F^H b and return f
    # Works for complex F, b; L is your first-difference matrix (real).
    FHF = F.conj().T @ F
    LTL = L.T @ L
    rhs = F.conj().T @ b
    K   = FHF + (lam**2) * LTL
    return spla.solve(K, rhs, assume_a='pos')

def solve_l1_cvx(A, L, g, mu):
    if not _HAS_CVXPY:
        raise RuntimeError("cvxpy not installed.")
    n = A.shape[1]
    f = cp.Variable(n)
    obj = cp.sum_squares(A @ f - g) + mu * cp.norm1(L @ f)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(verbose=False, solver=cp.OSQP)  # fallback solver, or try SCS
    return f.value




# Regularization parameters

class TikhonovNECache:
    # Complex-safe cache for Tikhonov normal equations.
    # Uses A^H A and A^H g; tries Cholesky, falls back to Hermitian indefinite solve.
    def __init__(self, A, L, g):
        self.A = A
        self.L = L
        self.g = g
        AH = A.conj().T
        LH = L.conj().T
        self.ATA = AH @ A
        self.LTL = LH @ L
        self.ATg = AH @ g

    def solve(self, lam):
        M = self.ATA + (lam**2) * self.LTL
        # Symmetrize to remove tiny asymmetry from roundoff
        M = (M + M.conj().T) * 0.5
        rhs = self.ATg
        try:
            c, lower = spla.cho_factor(M, overwrite_a=False, check_finite=False)
            f = spla.cho_solve((c, lower), rhs, check_finite=False)
        except spla.LinAlgError:
            # Hermitian (possibly indefinite) fallback (LDL^T / Bunch-Kaufman)
            f = spla.solve(M, rhs, assume_a='her', check_finite=False)
        return f

    def resid(self, f):
        return npl.norm(self.A @ f - self.g)

def pick_lambda_discrepancy_bisect(A, L, g, target_resid, lam_lo=1e-6, lam_hi=1e3, iters=14, tol=0.02):
    ## Morozov with log-bisection using a cached Cholesky-based solver.
    ## tol is relative on the residual (2% by default).
    cache = TikhonovNECache(A, L, g)

    def resid(lam):
        f = cache.solve(lam)
        return cache.resid(f), f

    r_lo, f_lo = resid(lam_lo)
    r_hi, f_hi = resid(lam_hi)

    # Expand to straddle target
    k = 0
    while r_lo > target_resid and lam_lo > 1e-12 and k < 6:
        lam_lo *= 0.3; r_lo, f_lo = resid(lam_lo); k += 1
    k = 0
    while r_hi < target_resid and lam_hi < 1e12 and k < 6:
        lam_hi *= 3.0; r_hi, f_hi = resid(lam_hi); k += 1

    lam_best, f_best = (lam_lo, f_lo) if abs(r_lo-target_resid) <= abs(r_hi-target_resid) else (lam_hi, f_hi)

    for _ in range(iters):
        lam_mid = 10**(0.5*(np.log10(lam_lo)+np.log10(lam_hi)))
        r_mid, f_mid = resid(lam_mid)
        if abs(r_mid-target_resid) < abs(cache.resid(f_best) - target_resid):
            lam_best, f_best = lam_mid, f_mid
        if abs(r_mid-target_resid) / (target_resid + 1e-18) < tol:
            break
        if r_mid < target_resid:
            lam_lo, r_lo = lam_mid, r_mid
        else:
            lam_hi, r_hi = lam_mid, r_mid

    return lam_best, f_best

class L1TVProblem:
    def __init__(self, A, L, g, solver="OSQP"):
        if not _HAS_CVXPY:
            raise RuntimeError("cvxpy not installed.")
        self.A, self.L, self.g = A, L, g
        self.n = A.shape[1]
        self.f = cp.Variable(self.n)
        self.mu = cp.Parameter(nonneg=True)
        self.obj = cp.sum_squares(A @ self.f - g) + self.mu * cp.norm1(L @ self.f)
        self.prob = cp.Problem(cp.Minimize(self.obj))
        self.solver = solver

    def solve_for(self, mu_value, **kw):
        self.mu.value = float(mu_value)
        self.prob.solve(solver=self.solver, warm_start=True,
                        eps_abs=1e-4, eps_rel=1e-4, verbose=False, **kw)
        return self.f.value

def pick_mu_discrepancy_bisect(A, L, g, target_resid, mu_lo=1e-3, mu_hi=1e1, iters=14):
    if not _HAS_CVXPY:
        raise RuntimeError("cvxpy not installed.")
    prob = L1TVProblem(A, L, g, solver="OSQP")

    def resid(mu):
        f = prob.solve_for(mu)
        return npl.norm(A @ f - g), f

    r_lo, f_lo = resid(mu_lo)
    r_hi, f_hi = resid(mu_hi)

    k = 0
    while r_lo > target_resid and mu_lo > 1e-8 and k < 6:
        mu_lo *= 0.3; r_lo, f_lo = resid(mu_lo); k += 1
    k = 0
    while r_hi < target_resid and mu_hi < 1e3 and k < 6:
        mu_hi *= 3.0; r_hi, f_hi = resid(mu_hi); k += 1

    if abs(r_lo - target_resid) <= abs(r_hi - target_resid):
        mu_best, f_best, gap_best = mu_lo, f_lo, abs(r_lo - target_resid)
    else:
        mu_best, f_best, gap_best = mu_hi, f_hi, abs(r_hi - target_resid)

    for _ in range(iters):
        mu_mid = 10**(0.5*(np.log10(mu_lo) + np.log10(mu_hi)))
        r_mid, f_mid = resid(mu_mid)
        gap = abs(r_mid - target_resid)
        if gap < gap_best:
            mu_best, f_best, gap_best = mu_mid, f_mid, gap
        if r_mid < target_resid:
            mu_lo, r_lo = mu_mid, r_mid
        else:
            mu_hi, r_hi = mu_mid, r_mid

    return mu_best, f_best




# Difference matrices (periodic) 
def diff_matrix_first(n):
    L = np.zeros((n, n))
    for i in range(n-1):
        L[i, i] = -1.0
        L[i, i+1] = 1.0
    L[n-1, n-1] = -1.0
    L[n-1, 0] = 1.0
    return L

def identity_matrix(n):
    return np.eye(n)





# Gaussian PSF & blur matrices 
def gaussian_psf_from_m_sigma(m=6, sigma=3.0):
    # Build symmetric kernel like assignment code
    tmp = np.linspace(0, 1, m+1)
    y = np.concatenate([-tmp[1:][::-1], tmp])
    psf = np.exp(-y**2/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    psf = psf / psf.sum()
    return psf

def blur_matrix_zero_boundary(psf, n):
    # Build n x n matrix A: (Af)[i] = (psf * f)[i] with zero-boundary
    L = len(psf)
    if L % 2 == 0:
        raise ValueError("Use an odd-length psf.")
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0
        y = np.convolve(e, psf, mode='same')
        A[:, j] = y
    return A

def blur_matrix_periodic(psf, n):
    # Periodic/circular convolution matrix via circulant embedding
    L = len(psf)
    if L % 2 == 0:
        raise ValueError("Use an odd-length psf.")
    m = (L - 1)//2
    h = np.zeros(n)
    h[0] = psf[m]                     # center
    for k in range(1, m+1):
        h[k]     = psf[m + k]         # positive shifts
        h[n - k] = psf[m - k]         # negative shifts
    # circulant needs first col c with c[0]=h[0], c[k]=h[n-k]
    c = np.r_[h[0], h[:0:-1]]
    C = spla.circulant(c)
    return C


def solve_once_morozov(A, f_true, L, rel, tau=1.05, want_tv=True, eta_dir=None):
    """
    LS, Tikhonov (L2), and TV (L1 on D1) with Morozov target = tau * ||e||.
    Returns reconstructions, errors, and chosen parameters.
    """
    g_clean = A @ f_true
    g_noisy, e, eta = add_relative_noise_fixed(g_clean, rel, eta_dir)
    target = tau * np.linalg.norm(e)

    # LS
    f_ls = solve_ls(A, g_noisy)

    # Tikhonov via discrepancy principle
    lam, f_tik = pick_lambda_discrepancy_bisect(A, L, g_noisy, target)

    out = {
        "eta": eta,
        "f_ls": f_ls, "f_tik": f_tik, "lam": lam,
        "err_ls": relerr(f_ls, f_true),
        "err_tik": relerr(f_tik, f_true)
    }

    # TV (if cvxpy available)
    try:
        has_cvx = _HAS_CVXPY
    except NameError:
        has_cvx = False
    if want_tv and has_cvx:
        try:
            mu, f_tv = pick_mu_discrepancy_bisect(A, L, g_noisy, target)
            out.update({"f_tv": f_tv, "mu": mu, "err_tv": relerr(f_tv, f_true)})
        except Exception as ex:
            print("TV skipped:", ex)
    return out


def make_A_zero(n, sigma):
    m_here = int(np.ceil(3*sigma))  # cover ~±3σ so you don't truncate the PSF
    psf_here = gaussian_psf_from_m_sigma(m=m_here, sigma=sigma)
    return blur_matrix_zero_boundary(psf_here, n)




# def plot_overlay(x, f_true, series, title):
#     """
#     series: list of (y, label, style_dict)
#     """
#     fig, ax = plt.subplots()
#     ax.plot(x, f_true, lw=2, label="True")
#     for y, lab, st in series:
#         ax.plot(x, y, label=lab, **(st or {}))
#     ax.set_title(title)
#     ax.set_xlabel("index"); ax.set_ylabel("amplitude")
#     ax.legend()
#     plt.show()






# Noise 
def add_relative_noise(g, rel=0.01):
    e = np.random.randn(*g.shape)
    e = rel * npl.norm(g) * (e / npl.norm(e))
    return g + e, e

def add_relative_noise_fixed(g, rel=0.01, eta=None, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    if eta is None:
        z = rng.standard_normal(g.shape)
        eta = z / (npl.norm(z) + 1e-15)
    e = rel * (npl.norm(g) + 1e-15) * eta
    return g + e, e, eta

def add_complex_noise(b, eta=0.1):
    # Complex Gaussian CN(0, eta^2 I): real/imag ~ N(0, eta^2/2)
    re = np.random.randn(*b.shape) * (eta/np.sqrt(2))
    im = np.random.randn(*b.shape) * (eta/np.sqrt(2))
    return b + (re + 1j*im), (re + 1j*im)







### Fourier Helpers

def piecewise_const_hat_k(k):
    # Vectorized Fourier coefficients for:
    # f(t)=1 on (-1/4, 0], 2 on [1/2, 7/8], else 0 (period 2 on [-1,1]).
    # Accepts scalar or array k (integers); returns same shape, complex dtype.
    k = np.asarray(k)
    out = np.zeros_like(k, dtype=np.complex128)

    # k = 0: average value over a full period
    mask0 = (k == 0)
    out[mask0] = 0.5

    # k != 0
    km = k[~mask0].astype(float)
    ikpi = 1j * km * np.pi

    # ∫_{-1/4}^{0} 1 · e^{-ikπt} dt
    term1 = (np.exp(0.0) - np.exp(1j * (km * np.pi / 4.0))) / (-ikpi)
    # ∫_{1/2}^{7/8} 2 · e^{-ikπt} dt
    term2 = 2.0 * (np.exp(-1j * (km * np.pi * 7.0 / 8.0)) - np.exp(-1j * (km * np.pi / 2.0))) / (-ikpi)

    out[~mask0] = 0.5 * (term1 + term2)

    # return scalar for scalar input
    if out.shape == ():
        return out.item()
    return out

def fourier_partial_sum(n, t):
    # modes k = -n/2,...,n/2
    ks = np.arange(-n//2, n//2 + 1)
    out = np.zeros_like(t, dtype=np.complex128)
    for k in ks:
        out += piecewise_const_hat_k(k) * np.exp(1j * k * np.pi * t)
    return out





# Filters

def raised_cosine_filter(k, N):
    # σ_RC(k) = 0.5*(1+cos(π k / N)) on |k|<=N, else 0
    sig = 0.5 * (1 + np.cos(np.pi * k / N))
    sig[np.abs(k) > N] = 0.0
    return sig

def exponential_filter(k, N, q=4):
    # σ_exp(k) = exp(-( |k|/N )^q )
    sig = np.exp(- (np.abs(k)/N)**q )
    sig[np.abs(k) > N] = 0.0
    return sig





def dft_matrix_assignment(n):
    # F(l,j) = (1/n) * exp( -i (l - n/2 - 1) π t_j ), with t_j = -1 + 2j/n, j=1..n, l=1..n
    j = np.arange(1, n+1)
    t = -1 + 2*j/n
    l = np.arange(1, n+1)
    K = (l - (n/2) - 1)[:, None]  # shape (n,1)
    F = (1/n) * np.exp(-1j * (K * np.pi) * t[None, :])
    return F, t



