"""
Reproduces all figures and Monte Carlo tables of Section 6 (Numerical
Experiments) of the manuscript "A weak stochastic maximum principle for
sparse stochastic optimal control with control entering drift and
diffusion":

  fig_feedback_laws.pdf       (Examples 1/3) bang-bang-off vs continuous
                              saturation law; switching signal pinned at -lambda
  fig_box_vs_ball.pdf         (Example 2)    box vs L1-ball: geometry and
                              ensemble activation statistics
  fig_sigma1_sweep.pdf        (Example 3)    diffusion-coupling diagnostic sweep
  fig_picard_convergence.pdf  (Example 1)    damped Picard convergence
  lambda sweep                (Table lambda-sweep) L^1-weight study,
                              printed to stdout

Evaluation protocols (matching the manuscript):
  * grid-feedback MC:     M = 5e4 paths, seed 7 (1D) / seed 5 (2D);
  * BBO family MC:        M = 2e4 paths, seed 11;
  * Picard out-of-sample: M' = 4e4 fresh paths (seed 123).

Revisions in this version:
  * fixed off-by-one in mc_eval_2d snapshot indexing (krow): the t = 0
    step previously read the never-written, all-zero last row of K1/K2;
  * the 2D HJB solver now stores the final policy in the last snapshot row;
  * Howard policy iteration uses a relative stopping criterion
    ||du||_inf / max(1, |U|) < 1e-10 instead of exact equality, which
    accelerates the control-dependent-diffusion solves by an order of
    magnitude without changing their output;
  * float64 is used consistently throughout the Picard solver;
  * added make_lambda_sweep for the L^1-weight study (Table lambda-sweep).
  * performance-audited optimized edition: exact drift-only BBO policy update,
    vectorized BBO-family interpolation, compact int8 2D policy snapshots,
    lower-memory Picard candidate evaluation, and reuse of canonical HJB solves.
  * corrected 2D feedback-snapshot time alignment: recompute the terminal-tau
    policy from the final value grid and use krow=(Nt-i)//store_every.

Everything is computed from scratch:
  * 1D HJB: implicit Euler in time + policy iteration (Howard), upwind
    finite differences, control mesh for the inner minimization;
  * 2D HJB: explicit upwind scheme in tau = T - t with the EXACT inner
    minimization for the box and the L1-ball (no control mesh);
  * closed-loop Monte Carlo evaluation of the converged feedback laws
    on independent Brownian paths;
  * damped Picard iteration with least-squares Monte Carlo
    (Gobet-Lemor-Warin) and binned policy projection for the coupled FBSDE.

Usage:
  python make_figures.py all        # everything
  python make_figures.py fig1       # only fig_feedback_laws.pdf
  python make_figures.py fig2       # only fig_box_vs_ball.pdf
  python make_figures.py fig3       # only fig_sigma1_sweep.pdf
  python make_figures.py fig4       # only fig_picard_convergence.pdf
  python make_figures.py lamsweep   # only the lambda-sweep table rows
  python make_figures.py refine2d   # only the 2D mesh-refinement rows

Figures are written to FIGDIR (default: ./figures next to this script).
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# ----------------------------------------------------------------------
# output directory and global plot style
# ----------------------------------------------------------------------
OUT_DIR = os.environ.get(
    "FIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.framealpha": 0.9,
})

# muted, warm-toned palette
C_BLUE = "#4C72B0"
C_RED = "#C44E52"
C_GREEN = "#55A868"
C_ORANGE = "#DD8452"
C_GRAY = "#8C8C8C"

# ----------------------------------------------------------------------
# model parameters (Examples 1 and 3)
# ----------------------------------------------------------------------
ALPHA, BETA, SIG0 = -0.5, 1.0, 0.2
LAM, CMAX, X0, T = 0.5, 2.0, 1.0, 1.0

# ----------------------------------------------------------------------
# high-resolution settings (choose reasonable values to balance accuracy)
# ----------------------------------------------------------------------
NX_1D = 2400        # 1D HJB space nodes
NT_1D = 1600        # 1D HJB time steps
NU_1D = 601         # control mesh size

NX_2D = 240         # 2D HJB space nodes (per dimension)
NT_2D = 4000        # 2D HJB time steps

MC_PATHS = 50000    # default Monte Carlo paths
PICARD_M = 50000    # paths used in Picard iteration
PICARD_N = 2000     # time steps in Picard
PICARD_NBINS = 1000 # state bins
PICARD_MAXSWEEPS = 40
PICARD_TOL = 1e-5
PICARD_THETA0 = 0.3

def exact_J0_scalar(alpha=ALPHA, sig0=SIG0, x0=X0, T=T):
    """Closed-form uncontrolled cost of the scalar benchmark (Eq. J(0))."""
    m2 = lambda s: np.exp(2 * alpha * s) * x0**2 \
        + sig0**2 / (2 * alpha) * (np.exp(2 * alpha * s) - 1.0)
    grid = np.linspace(0, T, 20001)
    run = 0.5 * np.trapezoid(m2(grid), grid)
    return run + 0.5 * m2(T)


# ======================================================================
# 1D HJB solver: implicit Euler + policy iteration (Howard)
# ======================================================================
def solve_hjb_1d_full(alpha=ALPHA, beta=BETA, sig0=SIG0, sig1=0.0,
                      lam=LAM, c=CMAX, T=T, L=6.0,
                      Nx=NX_1D, Nt=NT_1D, Nu=NU_1D, max_howard=30, verbose=False):
    """
    Solve  min_u { (alpha x + beta u) V_x + 1/2 (sig0+sig1 u)^2 V_xx
                   + 1/2 x^2 + lam |u| } + V_t = 0,  V(T,x) = 1/2 x^2
    on [-L, L] with Dirichlet data V = 1/2 x^2 at the boundary.

    Performance notes
    -----------------
    * If sig1 == 0, the control-mesh minimization is exactly equivalent to the
      bang-bang-off rule u in {-c,0,c}; because the uniform odd mesh contains
      those three values, this shortcut is lossless relative to the original.
    * If sig1 != 0, the original control-mesh protocol is retained, but all
      control-only terms are precomputed and only one full argmin pass is used.
    """
    x = np.linspace(-L, L, Nx)
    dx = x[1] - x[0]
    dt = T / Nt
    n = Nx - 2
    xi = x[1:-1]
    bc = 0.5 * np.array([x[0]**2, x[-1]**2])
    ugrid = np.linspace(-c, c, Nu)
    f_run = 0.5 * xi**2

    # The shortcut is lossless only when the original mesh contains u=0.
    drift_only = (abs(sig1) < 1e-15 and Nu % 2 == 1
                  and abs(ugrid[Nu // 2]) < 1e-14)
    if not drift_only:
        beta_u = beta * ugrid
        diff_u = 0.5 * (sig0 + sig1 * ugrid)**2
        l1_u = lam * np.abs(ugrid)
        i0 = Nu // 2
        row_idx = np.arange(n)

    W = 0.5 * x**2
    U = np.zeros(Nx)
    V_snap = np.empty((Nt + 1, Nx))
    U_snap = np.empty((Nt + 1, Nx))
    V_snap[0], U_snap[0] = W, U

    ab = np.empty((3, n))
    W_new = np.empty(Nx)
    U_new = np.empty(Nx)
    U_new[[0, -1]] = 0.0

    for k in range(Nt):
        W_old = W.copy()
        for _ in range(max_howard):
            Ui = U[1:-1]
            a = alpha * xi + beta * Ui
            D = 0.5 * (sig0 + sig1 * Ui)**2
            main = 1.0 + dt * (2.0 * D / dx**2 + np.abs(a) / dx)
            upper = -dt * (D / dx**2 + np.maximum(a, 0.0) / dx)
            lower = -dt * (D / dx**2 + np.maximum(-a, 0.0) / dx)
            rhs = W_old[1:-1] + dt * (f_run + lam * np.abs(Ui))
            rhs[0] += dt * (D[0] / dx**2 + max(-a[0], 0.0) / dx) * bc[0]
            rhs[-1] += dt * (D[-1] / dx**2 + max(a[-1], 0.0) / dx) * bc[1]

            ab.fill(0.0)
            ab[0, 1:] = upper[:-1]
            ab[1] = main
            ab[2, :-1] = lower[1:]
            W_new[[0, -1]] = bc
            W_new[1:-1] = solve_banded((1, 1), ab, rhs,
                                        overwrite_ab=False, overwrite_b=False,
                                        check_finite=False)

            Wx = (W_new[2:] - W_new[:-2]) / (2 * dx)
            if drift_only:
                # Exact minimizer of beta*Wx*u + lam*|u| on [-c,c].
                q = beta * Wx
                umin = np.zeros_like(q)
                umin[q > lam] = -c
                umin[q < -lam] = c
            else:
                Wxx = (W_new[2:] - 2 * W_new[1:-1] + W_new[:-2]) / dx**2

                # Exact O(Nx) minimization over the SAME uniform control mesh.
                # On u>=0 and u<=0 the Hamiltonian is quadratic:
                #   A u^2 + (B +/- lam) u + const,
                # so a discrete minimum lies at a side endpoint or at one of
                # the two mesh points bracketing the continuous stationary point.
                # We still evaluate the ORIGINAL precomputed cost terms at those
                # mesh indices and preserve np.argmin's lowest-index tie rule,
                # followed by the manuscript's special 0,+c,-c tolerance rule.
                A2 = (sig1 * sig1) * Wxx          # = 2*A
                B0 = beta * Wx + sig0 * sig1 * Wxx
                h_u = (2.0 * c) / (Nu - 1)

                convex = A2 > 0.0
                # stationary point on negative half: A2*u + (B0-lam)=0
                un = np.zeros_like(Wx)
                up = np.zeros_like(Wx)
                np.divide(-(B0 - lam), A2, out=un, where=convex)
                np.divide(-(B0 + lam), A2, out=up, where=convex)
                un = np.clip(un, -c, 0.0)
                up = np.clip(up, 0.0, c)

                jn = (un + c) / h_u
                jp = (up + c) / h_u
                jnf = np.floor(jn).astype(np.int64)
                jnc = np.ceil(jn).astype(np.int64)
                jpf = np.floor(jp).astype(np.int64)
                jpc = np.ceil(jp).astype(np.int64)
                jnf = np.clip(jnf, 0, i0)
                jnc = np.clip(jnc, 0, i0)
                jpf = np.clip(jpf, i0, Nu - 1)
                jpc = np.clip(jpc, i0, Nu - 1)

                # For non-convex/linear side problems, endpoints suffice.
                jnf = np.where(convex, jnf, 0)
                jnc = np.where(convex, jnc, i0)
                jpf = np.where(convex, jpf, i0)
                jpc = np.where(convex, jpc, Nu - 1)

                cand = np.column_stack((
                    np.zeros(n, dtype=np.int64),
                    jnf, jnc,
                    np.full(n, i0, dtype=np.int64),
                    jpf, jpc,
                    np.full(n, Nu - 1, dtype=np.int64),
                ))
                cand.sort(axis=1)  # reproduces full-grid argmin tie ordering
                ccost = (Wx[:, None] * beta_u[cand]
                         + Wxx[:, None] * diff_u[cand]
                         + l1_u[cand])
                kbest = np.argmin(ccost, axis=1)
                amin = cand[row_idx, kbest]
                cmin = ccost[row_idx, kbest]
                umin = ugrid[amin]

                # Preserve original tie-breaking: 0, then +c, then -c.
                tol_tie = 1e-9 * (1.0 + np.abs(cmin))
                cost0 = Wx * beta_u[i0] + Wxx * diff_u[i0] + l1_u[i0]
                costPc = Wx * beta_u[-1] + Wxx * diff_u[-1] + l1_u[-1]
                costNc = Wx * beta_u[0] + Wxx * diff_u[0] + l1_u[0]
                prefer0 = cost0 <= cmin + tol_tie
                umin = np.where(prefer0, 0.0, umin)
                preferPc = costPc <= cmin + tol_tie
                umin = np.where(~prefer0 & preferPc, c, umin)
                preferNc = costNc <= cmin + tol_tie
                umin = np.where(~prefer0 & ~preferPc & preferNc, -c, umin)

            U_new[1:-1] = umin
            du = np.max(np.abs(U_new - U) / np.maximum(1.0, np.abs(U)))
            # swap buffers rather than allocating new state arrays
            W, W_new = W_new, W
            U, U_new = U_new, U
            U_new[[0, -1]] = 0.0
            if du < 1e-10:
                break

        V_snap[k + 1], U_snap[k + 1] = W, U
        if verbose and (k + 1) % max(Nt // 5, 1) == 0:
            print(f"    HJB tau-step {k+1}/{Nt}")

    V0 = np.interp(X0, x, V_snap[-1])
    return dict(x=x, dt=dt, V=V_snap, U=U_snap, V0=V0,
                sig1=sig1, lam=lam, c=c)


def mc_eval_1d(sol, M=MC_PATHS, alpha=ALPHA, beta=BETA, sig0=SIG0,
               lam=LAM, x0=X0, T=T, seed=7, law=None):
    """
    Closed-loop Monte Carlo of the stored feedback (or of a callable
    law(tau_index, x) overriding it) on M independent paths.
    Returns (mean, se, off_time, interior_share, U_paths_mean|None).
    """
    x, V, U = sol["x"], sol["V"], sol["U"]
    Nt = V.shape[0] - 1
    dt = sol["dt"]
    sig1 = sol["sig1"]
    rng = np.random.default_rng(seed)
    dW = rng.normal(0.0, np.sqrt(dt), size=(M, Nt))
    X = np.full(M, x0)
    cost = np.zeros(M)
    off = np.zeros(M)
    act_interior = np.zeros(M)
    act_total = np.zeros(M)
    c = sol["c"]
    for i in range(Nt):
        row = Nt - i                     # V(t_i) = V_snap[Nt - i]
        if law is None:
            # nearest-neighbour evaluation of the grid feedback law
            ix = np.clip(np.searchsorted(x, X), 1, len(x) - 1)
            ix = np.where(np.abs(X - x[ix - 1]) < np.abs(X - x[ix]),
                          ix - 1, ix)
            u = U[row][ix]
        else:
            u = law(row, X)
        a = np.abs(u)
        cost += (0.5 * X**2 + lam * a) * dt
        off += (a < 1e-12)
        act_total += (a >= 1e-12)
        act_interior += (a > 0.01 * c) & (a < 0.99 * c)
        X = X + (alpha * X + beta * u) * dt + (sig0 + sig1 * u) * dW[:, i]
    cost += 0.5 * X**2
    off /= Nt
    interior_share = act_interior.sum() / max(act_total.sum(), 1.0)
    return cost.mean(), cost.std() / np.sqrt(M), off.mean(), interior_share


# ======================================================================
# Figure 1: optimal feedback laws (bang-bang-off vs continuous saturation)
# ======================================================================
def make_fig1(sol0=None, sol3=None):
    print("[fig1] fine HJB solves (sigma1 = 0 and 0.3) ...")
    t0 = time.time()
    if sol0 is None:
        sol0 = solve_hjb_1d_full(sig1=0.0)
    if sol3 is None:
        sol3 = solve_hjb_1d_full(sig1=0.3)
    print(f"    V_FD(0,1): sigma1=0 -> {sol0['V0']:.4f}, "
          f"sigma1=0.3 -> {sol3['V0']:.4f}   ({time.time()-t0:.0f}s)")

    x = sol0["x"]
    dx = x[1] - x[0]
    t_snap = 0.5
    row = int(round((T - t_snap) / sol0["dt"]))

    u0 = sol0["U"][row]
    u3 = sol3["U"][row]
    V3 = sol3["V"][row]
    Wx = np.gradient(V3, dx)
    Wxx = np.gradient(Wx, dx)
    psi = -BETA * Wx - 0.3 * (SIG0 + 0.3 * u3) * Wxx

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    a = ax[0]
    a.plot(x, u0, color=C_BLUE, lw=1.6,
           label=r"$\sigma_1=0$ (drift-only): bang-bang-off")
    a.plot(x, u3, color=C_GREEN, lw=1.6,
           label=r"$\sigma_1=0.3$: continuous saturation")
    a.axhline(-CMAX, color=C_GRAY, lw=0.8, ls=":")
    a.axhline(CMAX, color=C_GRAY, lw=0.8, ls=":")
    a.set_xlim(-1.3, 1.3); a.set_ylim(-2.4, 2.4)
    a.set_xlabel(r"$x$"); a.set_ylabel(r"$u^*(t_0,x)$")
    a.set_title(r"(a) Optimal feedback at $t_0=0.5$, $\lambda=0.5$ (zoom)")
    a.legend(loc="upper right", fontsize=8)

    b = ax[1]
    b.plot(x, psi, color=C_RED, lw=1.6, label=r"$\psi(t_0,x)$, $\sigma_1=0.3$")
    b.axhline(LAM, color="k", lw=1.0, ls="--")
    b.axhline(-LAM, color="k", lw=1.0, ls="--")
    b.axhspan(-LAM, LAM, color=C_GRAY, alpha=0.10)
    b.set_xlim(-1.3, 1.3); b.set_ylim(-1.5, 1.1)
    b.set_xlabel(r"$x$"); b.set_ylabel(r"$\psi(t_0,x)$")
    b.set_title(r"(b) Signal pinned at $-\lambda$ on the ramp")
    b.annotate("ramp: $\\psi=-\\lambda$, $u^*$ interior",
               xy=(0.56, -LAM), xytext=(-1.15, -1.15), fontsize=8,
               arrowprops=dict(arrowstyle="->", lw=0.8))
    b.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_feedback_laws.pdf")
    fig.savefig(path); plt.close(fig)
    print(f"    saved {path}")
    return sol0, sol3


# ======================================================================
# Damped Picard + LSMC solver for the coupled FBSDE (Example 1)
# ======================================================================
def fbsde_picard_scalar(alpha=ALPHA, beta=BETA, sig0=SIG0, lam=LAM, c=CMAX,
                        x0=X0, T=T, N=PICARD_N, M=PICARD_M, n_bins=PICARD_NBINS,
                        bin_lo=-2.5, bin_hi=3.5, theta0=PICARD_THETA0,
                        max_sweeps=PICARD_MAXSWEEPS, tol=PICARD_TOL, plateau=3,
                        seed=1, verbose=True, out_M=40000):
    """Damped Picard + LSMC with lower-memory candidate evaluation.

    The numerical protocol is unchanged.  The candidate bang-bang-off array is
    stored as int8 signs (lossless), trial trajectories retain only the current
    state, and projection histograms are formed only for the accepted trial.
    """
    dt = T / N
    sqrt_dt = np.sqrt(dt)
    rng = np.random.default_rng(seed)
    dW = rng.normal(0.0, sqrt_dt, size=(M, N)).astype(np.float64)
    edges = np.linspace(bin_lo, bin_hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_scale = n_bins / (bin_hi - bin_lo)
    policy = np.zeros((N + 1, n_bins), dtype=np.float64)

    def bin_index(x):
        # Uniform-grid equivalent of digitize(...)-1 for non-edge states.
        return np.clip(((x - bin_lo) * bin_scale).astype(np.int64),
                       0, n_bins - 1)

    def basis(x):
        return np.column_stack([np.ones_like(x), x, x**2, x**3, x**4])

    def simulate_binned(pol, dW_, stats=False, store_paths=True):
        Mm, Nn = dW_.shape
        if store_paths:
            X = np.empty((Mm, Nn + 1), dtype=np.float64)
            X[:, 0] = x0
            xcur = X[:, 0]
        else:
            X = None
            xcur = np.full(Mm, x0, dtype=np.float64)
        cst = np.zeros(Mm)
        off = np.zeros(Mm)
        act = np.zeros(Mm)
        full = np.zeros(Mm)
        for i in range(Nn):
            idx = bin_index(xcur)
            u = pol[i, idx]
            cst += (0.5 * xcur**2 + lam * np.abs(u)) * dt
            if stats:
                on = np.abs(u) > 1e-12
                off += ~on; act += on
                full += on & (np.abs(np.abs(u) - c) < 1e-6)
            xnext = xcur + (alpha * xcur + beta * u) * dt + sig0 * dW_[:, i]
            if store_paths:
                X[:, i + 1] = xnext
                xcur = X[:, i + 1]
            else:
                xcur = xnext
        cst += 0.5 * xcur**2
        return X, cst, off / Nn, full / np.maximum(act, 1)

    def eval_candidate(Uc_code, theta, pol, collect=False):
        Mm, Nn = dW.shape
        xcur = np.full(Mm, x0, dtype=np.float64)
        cst = np.zeros(Mm)
        if collect:
            sums = np.zeros((Nn + 1, n_bins), dtype=np.float64)
            cnts = np.zeros((Nn + 1, n_bins), dtype=np.int64)
        else:
            sums = cnts = None
        omt = 1.0 - theta
        for i in range(Nn):
            idx = bin_index(xcur)
            uc = c * Uc_code[:, i].astype(np.float64)
            u = omt * pol[i, idx] + theta * uc
            cst += (0.5 * xcur**2 + lam * np.abs(u)) * dt
            if collect:
                sums[i] += np.bincount(idx, weights=u, minlength=n_bins)
                cnts[i] += np.bincount(idx, minlength=n_bins)
            xcur = xcur + (alpha * xcur + beta * u) * dt + sig0 * dW[:, i]
        cst += 0.5 * xcur**2
        return cst.mean(), sums, cnts

    X, cst, _, _ = simulate_binned(policy, dW, store_paths=True)
    J = cst.mean()
    history = []
    stall = 0
    for k in range(1, max_sweeps + 1):
        # int8 is lossless because the candidate takes only {-c,0,c}.
        Uc_code = np.empty((M, N), dtype=np.int8)
        p = -X[:, -1]
        for i in range(N - 1, -1, -1):
            y = p + (alpha * p - X[:, i + 1]) * dt
            A = basis(X[:, i])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            p = A @ coef
            Uc_code[:, i] = np.where(p > lam / beta, 1,
                                     np.where(p < -lam / beta, -1, 0)).astype(np.int8)

        # Evaluate trial costs without allocating projection tables.
        accepted_theta = None
        accepted_J = None
        theta = theta0
        while theta > 1e-3:
            Jt, _, _ = eval_candidate(Uc_code, theta, policy, collect=False)
            if Jt < J - 1e-6:
                accepted_theta, accepted_J = theta, Jt
                break
            theta *= 0.5

        Jp, _, _ = eval_candidate(Uc_code, 1.0, policy, collect=False)
        if Jp < J - 1e-6 and (accepted_J is None or Jp < accepted_J):
            accepted_theta, accepted_J = 1.0, Jp

        if accepted_theta is None:
            stall += 1
            history.append(J)
            if verbose:
                print(f"    sweep {k}: rejected, J stays {J:.4f}", flush=True)
            if stall >= plateau:
                break
            continue

        # One deterministic replay of the selected candidate builds projection.
        _, sums, cnts = eval_candidate(Uc_code, accepted_theta, policy, collect=True)
        policy = np.where(cnts > 0, sums / np.maximum(cnts, 1), policy).astype(np.float64)
        X, cst, _, _ = simulate_binned(policy, dW, store_paths=True)
        Jn = cst.mean()
        rel = abs(J - Jn) / max(abs(J), 1e-12)
        history.append(Jn)
        if verbose:
            print(f"    sweep {k}: J = {Jn:.4f}", flush=True)
        stall = stall + 1 if rel < tol else 0
        J = Jn
        if stall >= plateau:
            break

    # Training arrays are no longer needed; release them before allocating the
    # fresh out-of-sample Brownian matrix.  This materially lowers peak RAM.
    del X, cst, dW
    try:
        del Uc_code
    except UnboundLocalError:
        pass

    rng2 = np.random.default_rng(123)
    dW2 = rng2.normal(0.0, sqrt_dt, size=(out_M, N)).astype(np.float64)
    _, cb, offb, fullb = simulate_binned(policy, dW2, stats=True, store_paths=False)
    pol_bbo = np.where(policy > 0.5 * c, c,
                       np.where(policy < -0.5 * c, -c, 0.0)).astype(np.float64)
    _, cp, offp, fullp = simulate_binned(pol_bbo, dW2, stats=True, store_paths=False)
    out = dict(binned=(cb.mean(), cb.std() / np.sqrt(len(cb)), offb.mean(), fullb.mean()),
               bbo=(cp.mean(), cp.std() / np.sqrt(len(cp)), offp.mean(), fullp.mean()))
    return dict(policy=policy, edges=edges, centers=centers, history=history, out=out)


def project_to_bbo(policy, c=CMAX):
    """Snap a binned policy onto the bang-bang-off class {-c, 0, c}."""
    out = np.zeros_like(policy)
    out[policy > 0.5 * c] = c
    out[policy < -0.5 * c] = -c
    return out


# ======================================================================
# Figure 4: damped Picard convergence
# ======================================================================
def make_fig4(sol=None):
    print("[fig4] damped Picard + LSMC for the coupled FBSDE ...")
    t0 = time.time()
    res = fbsde_picard_scalar()
    hist = res["history"]
    print(f"    converged in {len(hist)} sweeps: "
          + ", ".join(f"{h:.4f}" for h in hist)
          + f"   ({time.time()-t0:.0f}s)")
    ob, op = res["out"]["binned"], res["out"]["bbo"]
    print(f"    out-of-sample binned : {ob[0]:.4f}+-{ob[1]:.4f} off={ob[2]:.3f} full-ampl={ob[3]:.3f}")
    print(f"    out-of-sample BBOproj: {op[0]:.4f}+-{op[1]:.4f} off={op[2]:.3f} full-ampl={op[3]:.3f}")

    if sol is None:
        sol = solve_hjb_1d_full(sig1=0.0)
    print(f"    V_FD(0,1) = {sol['V0']:.4f}")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sweeps = np.arange(1, len(hist) + 1)
    ax.plot(sweeps, hist, "o-", color=C_BLUE, lw=1.6, ms=5,
            label="damped Picard (in-sample)")
    ax.axhline(sol["V0"], color=C_RED, ls="--", lw=1.4,
               label=f"finite-difference HJB benchmark ({sol['V0']:.4f})")
    ax.set_xlabel("Picard sweep $k$")
    ax.set_ylabel(r"in-sample cost $\widehat J^{(k)}$")
    ax.set_xticks(sweeps)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_picard_convergence.pdf")
    fig.savefig(path); plt.close(fig)
    print(f"    saved {path}")
    return sol


# ======================================================================
# Figure 3: diffusion-coupling diagnostic sweep (Example 3)
# ======================================================================
def bbo_family_mc(sol, thetas, M=20000, alpha=ALPHA, beta=BETA, sig0=SIG0,
                  lam=LAM, x0=X0, T=T, seed=11):
    """Vectorized bang-bang-off threshold-family Monte Carlo.

    The old inner Python loop interpolated each threshold column separately.
    Because every column uses the same x-grid and V_x snapshot, flattening X
    permits one np.interp call per time step with identical interpolation law.
    """
    x, V = sol["x"], sol["V"]
    Nt = V.shape[0] - 1
    dt = sol["dt"]
    sig1, c = sol["sig1"], sol["c"]
    dV = np.gradient(V, x[1] - x[0], axis=1)
    nth = len(thetas)
    rng = np.random.default_rng(seed)
    dW = rng.normal(0.0, np.sqrt(dt), size=(M, Nt))
    X = np.full((M, nth), x0)
    cost = np.zeros((M, nth))
    th = np.asarray(thetas)[None, :]
    for i in range(Nt):
        vx = np.interp(X.ravel(), x, dV[Nt - i]).reshape(M, nth)
        u = np.where(np.abs(vx) > th, -c * np.sign(vx), 0.0)
        cost += (0.5 * X**2 + lam * np.abs(u)) * dt
        X = X + (alpha * X + beta * u) * dt + (sig0 + sig1 * u) * dW[:, i:i+1]
    cost += 0.5 * X**2
    means = cost.mean(axis=0)
    j = int(np.argmin(means))
    return means[j], thetas[j], means


def make_fig3(precomputed=None):
    """Diffusion-coupling diagnostic sweep with optional HJB reuse."""
    print("[fig3] sigma1 sweep (fine HJB mesh Nx=2400, Nt=1600, Nu=601) ...")
    sig1s = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    thetas = np.linspace(0.05, 1.5, 30)
    precomputed = {} if precomputed is None else dict(precomputed)
    rows = []
    for s1 in sig1s:
        t0 = time.time()
        sol = precomputed.get(s1)
        if sol is None:
            sol = solve_hjb_1d_full(sig1=s1)
        mc, se, off, interior = mc_eval_1d(sol, M=MC_PATHS)
        bbo, th_best, _ = bbo_family_mc(sol, thetas, M=20000)
        gap = (bbo - mc) / mc * 100.0
        rows.append(dict(sig1=s1, Vfd=sol["V0"], mc=mc, se=se, off=off,
                         interior=interior, bbo=bbo, gap=gap))
        print(f"    sigma1={s1:.1f}: V_FD={sol['V0']:.4f}  MC={mc:.4f}"
              f"+-{se:.4f}  off={off:.2f}  interior={interior:.2f}"
              f"  bestBBO={bbo:.4f} (gap {gap:.1f}%)  [{time.time()-t0:.0f}s]")

    S = np.array([r["sig1"] for r in rows]); VFD = np.array([r["Vfd"] for r in rows])
    MC = np.array([r["mc"] for r in rows]); SE = np.array([r["se"] for r in rows])
    OFF = np.array([r["off"] for r in rows]); INT = np.array([r["interior"] for r in rows])
    BBO = np.array([r["bbo"] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    a = ax[0]
    a.plot(S, VFD, "s-", color=C_GRAY, lw=1.4, ms=4, label="$V^{FD}$ (HJB)")
    a.errorbar(S, MC, yerr=2 * SE, fmt="o-", color=C_BLUE, lw=1.6, ms=4,
               label="optimal law (MC)")
    a.plot(S, BBO, "^--", color=C_RED, lw=1.4, ms=4, label="best bang-bang-off")
    a.set_xlabel(r"$\sigma_1$"); a.set_ylabel("cost")
    a.set_title("(a) Cost versus diffusion coupling"); a.legend(fontsize=8)
    b = ax[1]
    b.plot(S, OFF, "o-", color=C_BLUE, lw=1.6, ms=4, label="off-time fraction")
    b.plot(S, INT, "s-", color=C_ORANGE, lw=1.6, ms=4, label="interior-amplitude share")
    b.set_xlabel(r"$\sigma_1$"); b.set_ylabel("fraction"); b.set_ylim(-0.03, 1.03)
    b.set_title("(b) Structure of the optimal law"); b.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_sigma1_sweep.pdf")
    fig.savefig(path); plt.close(fig)
    print(f"    saved {path}")
    return rows


def make_lambda_sweep(lams=(0.1, 0.3, 0.5, 1.0), M=MC_PATHS, precomputed=None):
    """L1-weight sweep; optionally reuse a canonical lambda=0.5 HJB solve."""
    J0 = exact_J0_scalar()
    precomputed = {} if precomputed is None else dict(precomputed)
    print(f"[tab] lambda sweep (fine HJB mesh, M={M}, seed 7) ...")
    for lam in lams:
        t0 = time.time()
        sol = precomputed.get(float(lam))
        if sol is None:
            sol = solve_hjb_1d_full(sig1=0.0, lam=lam)
        mc, se, off, _ = mc_eval_1d(sol, M=M, lam=lam)
        print(f"    lam={lam:.1f}: V_FD={sol['V0']:.4f}  "
              f"MC={mc:.4f}+-{se:.4f}  off={off:.2f}  "
              f"improvement={(1.0 - mc / J0) * 100:.1f}%  "
              f"[{time.time() - t0:.0f}s]")


# ======================================================================
# 2D HJB solver (drift-only control, exact inner law) + closed-loop MC
# ======================================================================
def _policy_2d_from_value(W, dx, lam1, lam2, constraint, cpar):
    """Compute the exact 2D inner minimizer from one value-grid snapshot."""
    Wx1 = np.zeros_like(W)
    Wx2 = np.zeros_like(W)
    Wx1[1:-1, 1:-1] = (W[2:, 1:-1] - W[:-2, 1:-1]) / (2 * dx)
    Wx2[1:-1, 1:-1] = (W[1:-1, 2:] - W[1:-1, :-2]) / (2 * dx)
    u1 = np.zeros_like(W)
    u2 = np.zeros_like(W)
    if constraint == "box":
        c1, c2 = cpar
        m1 = np.abs(Wx1) > lam1
        m2 = np.abs(Wx2) > lam2
        u1[m1] = -c1 * np.sign(Wx1[m1])
        u2[m2] = -c2 * np.sign(Wx2[m2])
    else:
        e1 = np.abs(Wx1) - lam1
        e2 = np.abs(Wx2) - lam2
        act1 = (e1 > e2) & (e1 > 0)
        act2 = (e2 >= e1) & (e2 > 0)
        u1[act1] = -cpar * np.sign(Wx1[act1])
        u2[act2] = -cpar * np.sign(Wx2[act2])
    return u1, u2


def solve_hjb_2d_driftonly(a1=-0.5, a2=-0.3, s1=0.2, s2=0.2,
                           lam1=0.3, lam2=0.6, T=1.0, xgrid=4.0,
                           Nx=NX_2D, Nt=NT_2D, constraint="box", cpar=None,
                           store_K=True, store_every=5, verbose=False,
                           compact_K=True):
    """2D HJB solver with lossless compact policy snapshots.

    K snapshots are signs in int8 when compact_K=True; amplitudes are stored in
    K_amp1/K_amp2 and reconstructed exactly by mc_eval_2d.  The final policy is
    recomputed from the final W grid instead of reusing the pre-update policy.
    """
    if cpar is None:
        cpar = (1.0, 1.0) if constraint == "box" else 2.0
    x = np.linspace(-xgrid, xgrid, Nx)
    dx = x[1] - x[0]
    dt = T / Nt
    X1, X2 = np.meshgrid(x, x, indexing="ij")
    W = 0.5 * (X1**2 + X2**2)
    f_run = 0.5 * (X1**2 + X2**2)
    bc = W.copy()

    if constraint == "box":
        amp1, amp2 = float(cpar[0]), float(cpar[1])
    else:
        amp1 = amp2 = float(cpar)

    n_store = Nt // store_every + 1
    Kdtype = np.int8 if compact_K else np.float64
    K1 = np.zeros((n_store, Nx, Nx), dtype=Kdtype)
    K2 = np.zeros((n_store, Nx, Nx), dtype=Kdtype)

    for k in range(Nt):
        u1, u2 = _policy_2d_from_value(W, dx, lam1, lam2, constraint, cpar)
        if store_K and k % store_every == 0:
            if compact_K:
                K1[k // store_every] = np.sign(u1).astype(np.int8)
                K2[k // store_every] = np.sign(u2).astype(np.int8)
            else:
                K1[k // store_every], K2[k // store_every] = u1, u2

        b1 = a1 * X1 + u1
        b2 = a2 * X2 + u2
        b1i = b1[1:-1, 1:-1]
        b2i = b2[1:-1, 1:-1]
        Wc = W[1:-1, 1:-1]
        Wx1_up = np.where(b1i > 0, W[2:, 1:-1] - Wc, Wc - W[:-2, 1:-1]) / dx
        Wx2_up = np.where(b2i > 0, W[1:-1, 2:] - Wc, Wc - W[1:-1, :-2]) / dx
        Wxx1 = (W[2:, 1:-1] - 2 * Wc + W[:-2, 1:-1]) / dx**2
        Wxx2 = (W[1:-1, 2:] - 2 * Wc + W[1:-1, :-2]) / dx**2
        W[1:-1, 1:-1] = Wc + dt * (
            b1i * Wx1_up + b2i * Wx2_up
            + 0.5 * s1**2 * Wxx1 + 0.5 * s2**2 * Wxx2
            + f_run[1:-1, 1:-1]
            + lam1 * np.abs(u1[1:-1, 1:-1]) + lam2 * np.abs(u2[1:-1, 1:-1]))
        W[0, :], W[-1, :], W[:, 0], W[:, -1] = bc[0, :], bc[-1, :], bc[:, 0], bc[:, -1]
        if verbose and (k + 1) % max(Nt // 4, 1) == 0:
            print(f"    2D HJB tau-step {k+1}/{Nt}, max|W|={np.abs(W).max():.3e}")

    if store_K:
        # True terminal-tau policy based on the final W grid.
        u1f, u2f = _policy_2d_from_value(W, dx, lam1, lam2, constraint, cpar)
        if compact_K:
            K1[-1] = np.sign(u1f).astype(np.int8)
            K2[-1] = np.sign(u2f).astype(np.int8)
        else:
            K1[-1], K2[-1] = u1f, u2f

    gx = (1.5 + xgrid) / dx
    gy = (0.5 + xgrid) / dx
    i0, j0 = int(gx), int(gy)
    tx, ty = gx - i0, gy - j0
    V0 = ((1 - tx) * (1 - ty) * W[i0, j0] + tx * (1 - ty) * W[i0 + 1, j0]
          + (1 - tx) * ty * W[i0, j0 + 1] + tx * ty * W[i0 + 1, j0 + 1])
    return dict(x=x, dx=dx, dt=dt, V0=V0, K1=K1, K2=K2,
                K_compact=compact_K, K_amp1=amp1, K_amp2=amp2,
                store_every=store_every, Nt=Nt, constraint=constraint)


def mc_eval_2d(sol, M=MC_PATHS, a1=-0.5, a2=-0.3, s1=0.2, s2=0.2,
               lam1=0.3, lam2=0.6, x0=(1.5, 0.5), T=1.0, seed=5,
               legacy_snapshot_index=False):
    """Closed-loop Monte Carlo of stored 2D feedback.

    legacy_snapshot_index=True reproduces the uploaded script's (Nt-i-1)//se
    selection for audit comparison.  The corrected default uses (Nt-i)//se.
    """
    x, dx, dt = sol["x"], sol["dx"], sol["dt"]
    Nt, se = sol["Nt"], sol["store_every"]
    K1, K2 = sol["K1"], sol["K2"]
    compact = sol.get("K_compact", False)
    amp1, amp2 = sol.get("K_amp1", 1.0), sol.get("K_amp2", 1.0)
    xg = x[-1]
    rng = np.random.default_rng(seed)
    X1 = np.full(M, x0[0]); X2 = np.full(M, x0[1])
    cost = np.zeros(M); off = np.zeros(M)
    act1_only = np.zeros(M); act2_only = np.zeros(M); act_both = np.zeros(M)
    t_prof = np.arange(Nt) * dt
    e_u1 = np.zeros(Nt); e_u2 = np.zeros(Nt); p_both = np.zeros(Nt); e_l1 = np.zeros(Nt)
    sqrt_dt = np.sqrt(dt)
    for i in range(Nt):
        if legacy_snapshot_index:
            krow = min((Nt - i - 1) // se, K1.shape[0] - 1)
        else:
            krow = min((Nt - i) // se, K1.shape[0] - 1)
        ix = np.clip(((X1 + xg) / dx).round().astype(int), 0, len(x) - 1)
        iy = np.clip(((X2 + xg) / dx).round().astype(int), 0, len(x) - 1)
        if compact:
            u1 = amp1 * K1[krow, ix, iy].astype(np.float64)
            u2 = amp2 * K2[krow, ix, iy].astype(np.float64)
        else:
            u1 = K1[krow, ix, iy]; u2 = K2[krow, ix, iy]
        a1u, a2u = np.abs(u1), np.abs(u2)
        cost += (0.5 * (X1**2 + X2**2) + lam1 * a1u + lam2 * a2u) * dt
        on1, on2 = a1u > 1e-12, a2u > 1e-12
        off += (~on1) & (~on2)
        act1_only += on1 & (~on2); act2_only += on2 & (~on1); act_both += on1 & on2
        e_u1[i] = a1u.mean(); e_u2[i] = a2u.mean(); p_both[i] = (on1 & on2).mean()
        e_l1[i] = (a1u + a2u).mean()
        X1 = X1 + (a1 * X1 + u1) * dt + s1 * rng.normal(0, sqrt_dt, M)
        X2 = X2 + (a2 * X2 + u2) * dt + s2 * rng.normal(0, sqrt_dt, M)
    cost += 0.5 * (X1**2 + X2**2)
    off /= Nt
    active = act1_only + act2_only + act_both
    l1_when_active = float(M * e_l1.sum() / max(active.sum(), 1.0)) if active.sum() > 0 else 0.0
    return dict(mean=cost.mean(), se=cost.std() / np.sqrt(M), off=off.mean(),
                both_share=act_both.sum() / max(active.sum(), 1),
                ch1_share=act1_only.sum() / max(active.sum(), 1),
                l1_active=l1_when_active, p_both_peak=float(p_both.max()),
                t=t_prof, e_u1=e_u1, e_u2=e_u2, p_both=p_both, e_l1=e_l1)


def exact_J0_2d(a1=-0.5, a2=-0.3, s1=0.2, s2=0.2,
                x0=(1.5, 0.5), T=1.0):
    """Closed-form uncontrolled cost of the two-channel system."""
    g = np.linspace(0, T, 20001)
    m1 = x0[0]**2 * np.exp(2 * a1 * g) \
        + s1**2 / (2 * a1) * (np.exp(2 * a1 * g) - 1)
    m2 = x0[1]**2 * np.exp(2 * a2 * g) \
        + s2**2 / (2 * a2) * (np.exp(2 * a2 * g) - 1)
    run = 0.5 * np.trapezoid(m1 + m2, g)
    return run + 0.5 * (m1[-1] + m2[-1])


# ======================================================================
# Figure 2: box versus L1-ball
# ======================================================================
def make_fig2():
    print("[fig2] 2D HJB solves (box and L1-ball) + closed-loop MC ...")
    t0 = time.time()
    sol_box = solve_hjb_2d_driftonly(constraint="box", cpar=(1.0, 1.0))
    sol_ball = solve_hjb_2d_driftonly(constraint="ball", cpar=2.0)
    print(f"    V_FD(0,x0): box={sol_box['V0']:.4f}, "
          f"ball={sol_ball['V0']:.4f}   ({time.time()-t0:.0f}s)")
    J0 = exact_J0_2d()
    print(f"    exact uncontrolled J(0) = {J0:.4f}")

    st_box = mc_eval_2d(sol_box)
    st_ball = mc_eval_2d(sol_ball)
    print(f"    box : MC={st_box['mean']:.4f}+-{st_box['se']:.4f} "
          f"off={st_box['off']:.3f} both={st_box['both_share']:.3f} "
          f"ch1={st_box['ch1_share']:.3f} L1|act={st_box['l1_active']:.2f} "
          f"peakP(both)={st_box['p_both_peak']:.3f}")
    print(f"    ball: MC={st_ball['mean']:.4f}+-{st_ball['se']:.4f} "
          f"off={st_ball['off']:.3f} both={st_ball['both_share']:.3f} "
          f"ch1={st_ball['ch1_share']:.3f} L1|act={st_ball['l1_active']:.2f} "
          f"peakP(both)={st_ball['p_both_peak']:.3f}")

    # MC statistics are self-contained; release the large policy-snapshot cubes
    # before plotting.  This is especially valuable in the two-solver figure.
    del sol_box, sol_ball

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # (a) box geometry at the frozen instant psi = (2.0, 1.3)
    a = ax[0, 0]
    sq = plt.Rectangle((-1, -1), 2, 2, fill=True, facecolor=C_BLUE,
                       alpha=0.12, edgecolor=C_BLUE, lw=1.5)
    a.add_patch(sq)
    a.plot(1, 1, "o", color=C_BLUE, ms=9)
    a.annotate(r"$u^*=(1,1)$", xy=(1, 1), xytext=(0.1, 1.55), fontsize=10,
               arrowprops=dict(arrowstyle="->", lw=0.8))
    a.set_xlim(-2.4, 2.4)
    a.set_ylim(-2.4, 2.4)
    a.set_xlabel("$u_1$")
    a.set_ylabel("$u_2$")
    a.set_title("(a) Box $[-1,1]^2$: corner, both channels on")
    a.set_aspect("equal")

    # (b) L1-ball geometry at the same instant
    b = ax[0, 1]
    dia = plt.Polygon([(2, 0), (0, 2), (-2, 0), (0, -2)], closed=True,
                      facecolor=C_RED, alpha=0.10, edgecolor=C_RED, lw=1.5)
    b.add_patch(dia)
    b.plot(2, 0, "o", color=C_RED, ms=9)
    b.annotate(r"$u^*=(2,0)$", xy=(2, 0), xytext=(0.6, 1.4), fontsize=10,
               arrowprops=dict(arrowstyle="->", lw=0.8))
    b.set_xlim(-2.4, 2.4)
    b.set_ylim(-2.4, 2.4)
    b.set_xlabel("$u_1$")
    b.set_ylabel("$u_2$")
    b.set_title(r"(b) $L^1$-ball $|u|_1 \leq 2$: vertex, one channel")
    b.set_aspect("equal")

    # (c) box law: ensemble statistics
    c = ax[1, 0]
    c.plot(st_box["t"], st_box["e_u1"], color=C_BLUE, lw=1.6,
           label=r"$\mathbb{E}|u_1|$")
    c.plot(st_box["t"], st_box["e_u2"], color=C_GREEN, lw=1.6,
           label=r"$\mathbb{E}|u_2|$")
    c.plot(st_box["t"], st_box["p_both"], color=C_GRAY, lw=1.4, ls="--",
           label=r"$P(\mathrm{both\ channels\ on})$")
    c.set_xlabel("$t$")
    c.set_title("(c) Box law: concurrent activation")
    c.legend(fontsize=8)

    # (d) ball law: ensemble statistics
    d = ax[1, 1]
    d.plot(st_ball["t"], st_ball["e_l1"], color=C_RED, lw=1.6,
           label=r"$\mathbb{E}\|u\|_1$")
    d.plot(st_ball["t"], st_ball["e_u1"], color=C_BLUE, lw=1.4,
           label=r"$\mathbb{E}|u_1|$")
    d.plot(st_ball["t"], st_ball["e_u2"], color=C_GREEN, lw=1.4,
           label=r"$\mathbb{E}|u_2|$")
    d.axhline(2.0, color="k", lw=0.9, ls=":")
    d.set_xlabel("$t$")
    d.set_title("(d) Ball law: one channel, budget saturated")
    d.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_box_vs_ball.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"    saved {path}")

    
# ======================================================================
# Table: two-dimensional mesh-refinement study (coarse meshes)
# ======================================================================
def make_refinement_study(meshes=((120, 2000), (180, 3000)), M=MC_PATHS):
    """
    Two-dimensional mesh-refinement study (coarse rows of Table
    tab:2d-mesh-refinement).  Solves the box and L1-ball HJB problems on
    each mesh and re-simulates the grid feedback in closed loop on M paths
    (seed 5, manuscript protocol).  The finest mesh (240, 4000) is produced
    by make_fig2 (box 0.8485 / ball 0.7520) and is not repeated here.
    Rows are printed in LaTeX table format, ready to paste.
    """
    print("[tab] 2D mesh-refinement study (coarse meshes) ...")
    for Nx, Nt in meshes:
        t0 = time.time()
        sb = solve_hjb_2d_driftonly(constraint="box", cpar=(1.0, 1.0),
                                    Nx=Nx, Nt=Nt)
        sl = solve_hjb_2d_driftonly(constraint="ball", cpar=2.0,
                                    Nx=Nx, Nt=Nt)
        mb = mc_eval_2d(sb, M=M)
        ml = mc_eval_2d(sl, M=M)
        print(f"    $({Nx}$, ${Nt})$ & ${sb['V0']:.4f}$ & "
              f"${mb['mean']:.4f}\\pm{mb['se']:.4f}$ & ${sl['V0']:.4f}$ & "
              f"${ml['mean']:.4f}\\pm{ml['se']:.4f}$ \\\\"
              f"   [{time.time()-t0:.0f}s]")

# ======================================================================
# main
# ======================================================================
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t_start = time.time()
    if which == "all":
        # Run Picard before retaining reusable HJB arrays; then reuse the two
        # canonical fine-grid solves across Figures 1/3/4 and lambda=0.5.
        sol0 = make_fig4()
        sol0, sol3 = make_fig1(sol0=sol0)
        make_fig3(precomputed={0.0: sol0, 0.3: sol3})
        make_lambda_sweep(precomputed={0.5: sol0})
        del sol0, sol3
        make_fig2()
        make_refinement_study()      # 2D mesh-refinement rows (Table tab:2d-mesh-refinement)
    else:
        if which == "fig1": make_fig1()
        elif which == "fig4": make_fig4()
        elif which == "fig3": make_fig3()
        elif which == "fig2": make_fig2()
        elif which == "lamsweep": make_lambda_sweep()
        elif which == "refine2d": make_refinement_study()
        else:
            raise SystemExit("usage: python make_figures_optimized.py [all|fig1|fig2|fig3|fig4|lamsweep|refine2d]")
    print(f"done in {time.time()-t_start:.0f}s; figures in {OUT_DIR}")
