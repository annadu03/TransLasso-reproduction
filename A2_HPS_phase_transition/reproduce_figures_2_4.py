#!/usr/bin/env python3
"""Reproduce Yang et al. (2025), Figures 2(a,b) and 4(a,b).

Run all four figures:
    python3 reproduce_figures_2_4.py

Run one figure or change the Monte Carlo sample count:
    python3 reproduce_figures_2_4.py --figures 2b --reps 10

The four experiments use the parameter grids in FIGURES below.
Theoretical curves: Theorem 3 (Figure 2) and Theorem 10 (Figure 4).
Simulation results: direct Monte Carlo, author-style bias+MC variance,
and the exact finite-sample conditional mean risk.

NOTE: lambda is the FEATURE scaling used by the author notebooks, not
necessarily the covariance-eigenvalue lambda in the paper's prose.
Here Sigma_source has eigenvalues lambda**2 and lambda**(-2), while
Sigma_target = I. See README.md.
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save images without requiring a GUI.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq


# ============================================================
# 1. Experiment settings -- edit only here to change the grid
# ============================================================
SIGMA = 0.5
SEED = 2026
DEFAULT_REPS = 40
DEFAULT_NOISE_TRIALS = 20
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

FIGURES = {
    "2a": {
        "settings": [(200, 100, 4.0, 0.0), (300, 100, 4.0, 0.0),
                     (200, 50, 4.0, 0.0), (300, 50, 4.0, 0.0)],
        "max_n1": 1000, "sim_step": 25, "theory_step": 10,
    },
    "2b": {
        "settings": [(300, 100, lam, 0.0) for lam in (10, 4, 2, 1)],
        "max_n1": 1000, "sim_step": 25, "theory_step": 10,
    },
    "4a": {
        "settings": [(300, 100, lam, 0.1) for lam in (10, 4, 2, 1)],
        "max_n1": 1000, "sim_step": 50, "theory_step": 5,
    },
    "4b": {
        "settings": [(300, 100, 4.0, mu) for mu in (0.35, 0.27, 0.20)],
        "max_n1": 6000, "sim_step": 100, "theory_step": 10,
    },
}
# Each setting is (n2, p, lambda, mu).


# ============================================================
# 2. High-dimensional theory
# ============================================================
def source_feature_scales(p, lam):
    """X1 = Z1 @ diag(scales); X2 = Z2 (Target covariance = I)."""
    if p % 2 != 0:
        raise ValueError("p must be even for the paired-spectrum experiment")
    return np.r_[np.full(p // 2, lam), np.full(p // 2, 1.0 / lam)]


def figure2_theory(n1, n2, p, lam):
    """Theorem 3: covariate shift only (beta1 = beta2).

    N = n1 + n2, d_j = eigenvalues(Sigma_source).
    alpha1 + alpha2 = 1 - p/N,
    alpha1 + (1/N)*sum(d_j*alpha1/(d_j*alpha1+alpha2)) = n1/N.
    Risk = sigma^2/N * sum(1/(d_j*alpha1+alpha2)).
    """
    N = n1 + n2
    if n1 == 0:
        return SIGMA**2 * p / (n2 - p)
    if lam == 1:
        return SIGMA**2 * p / (N - p)

    eigvals = source_feature_scales(p, lam)**2
    total = 1.0 - p / N

    def equation(alpha1):
        alpha2 = total - alpha1
        return (alpha1 + np.sum(eigvals * alpha1 /
                                (eigvals * alpha1 + alpha2)) / N - n1 / N)

    alpha1 = brentq(equation, 0, total)
    alpha2 = total - alpha1
    return float(SIGMA**2 / N * np.sum(1 / (eigvals * alpha1 + alpha2)))


def figure4_theory(n1, n2, p, lam, mu):
    """Theorem 10: both covariate shift and model shift.

    Compute the root y0 and the auxiliary f1, f2, f3, then return
    (variance, bias). Specialised to the author's paired spectrum.
    """
    if n1 == 0:
        return SIGMA**2 * p / (n2 - p), 0.0

    n1, n2, p = float(n1), float(n2), float(p)
    q = lam
    # Cubic equation from the author's covariate-and-model-shift notebook.
    coefficients = [
        n2*q**2/n1,
        (2*n2-p)*(q**4+1)/(2*n1) + (p/n1-1)*q**2,
        (n2-p)*q**2/n1 + (p/(2*n1)-1)*(q**4+1),
        -q**2,
    ]
    roots = np.roots(coefficients)
    positive_roots = [float(z.real) for z in roots
                      if abs(z.imag) < 1e-8 and z.real > 0]
    if len(positive_roots) != 1:
        raise ArithmeticError("Theorem 10 root equation has no unique positive root")
    y = positive_roots[0]

    # g0(y) and its derivative (Sigma_target=I).
    g0 = p/(2*n1) * (q**2/(1+q**2*y) + 1/(q**2+y)) - 1/y
    g0_prime = -p/(2*n1) * (q**4/(1+q**2*y)**2
                              + 1/(q**2+y)**2) + 1/y**2
    f1 = n1*y/p + (n1-p)/(p*g0)
    f2 = n1/(p*g0_prime) - (n1-p)/(p*g0**2)
    f3 = -g0

    variance = SIGMA**2 * p / n1 * f1
    bias = 2*mu**2 * (1 - 2*f1*f3 + f2*f3**2) / (1 - p/n2*f2*f3**2)
    return float(variance), float(bias)


def theoretical_risk(fig, n1, n2, p, lam, mu):
    if fig.startswith("2"):
        return figure2_theory(n1, n2, p, lam)
    variance, bias = figure4_theory(n1, n2, p, lam, mu)
    return variance + bias


# ============================================================
# 3. Finite-dimensional Monte Carlo simulation
# ============================================================
def simulate_one_setting(n1_grid, n2, p, lam, mu, reps, noise_trials, rng):
    """Estimate Target excess risk on the same X for three methods.

    Direct: fit HPS on Y1=X1 beta1+noise and Y2=X2 beta2+noise.
    Author: E_beta[bias|X] + simulated noise variance.
    Finite exact: E_{beta,noise}[risk|X], calculated exactly.

    Since Sigma_target = I, Target excess risk is ||beta_hat-beta2||^2.
    """
    scales = source_feature_scales(p, lam)
    eye = np.eye(p)
    max_n1 = max(n1_grid)
    records = {method: np.zeros((reps, len(n1_grid))) for method in
               ("direct", "author", "finite_exact", "ols_direct", "ols_conditional")}

    for r in range(reps):
        # Generate one Target data set per outer repetition.
        X2 = rng.normal(size=(n2, p))
        S2 = X2.T @ X2
        beta2 = rng.normal(scale=mu / np.sqrt(p), size=p)
        beta1 = rng.normal(scale=mu / np.sqrt(p), size=p)
        eps2 = rng.normal(scale=SIGMA, size=n2)
        y2 = X2 @ beta2 + eps2

        fac2 = cho_factor(S2, check_finite=False)
        beta_ols = cho_solve(fac2, X2.T @ y2, check_finite=False)
        ols_direct = np.sum((beta_ols - beta2)**2)
        ols_conditional = SIGMA**2 * np.trace(
            cho_solve(fac2, eye, check_finite=False))

        # Nested Source data: increasing n1 uses more rows of the same X1.
        X1_full = rng.normal(size=(max_n1, p)) * scales
        y1_full = X1_full @ beta1 + rng.normal(scale=SIGMA, size=max_n1)

        # Independent noise for the author-style variance Monte Carlo.
        noise1 = rng.normal(scale=SIGMA, size=(max_n1, noise_trials))
        noise2 = rng.normal(scale=SIGMA, size=(n2, noise_trials))
        E2 = X2.T @ noise2
        S1 = np.zeros((p, p))
        c1 = np.zeros(p)
        E1 = np.zeros((p, noise_trials))
        previous_n1 = 0

        for j, n1 in enumerate(n1_grid):
            if n1 > previous_n1:
                Xnew = X1_full[previous_n1:n1]
                S1 += Xnew.T @ Xnew
                c1 += Xnew.T @ y1_full[previous_n1:n1]
                E1 += Xnew.T @ noise1[previous_n1:n1]
                previous_n1 = n1

            # HPS estimate: (S1+S2)^(-1) (X1'Y1+X2'Y2)
            fac = cho_factor(S1 + S2, check_finite=False)
            beta_hps = cho_solve(fac, c1 + X2.T @ y2, check_finite=False)
            K = cho_solve(fac, eye, check_finite=False)

            # Bias = 2 mu^2 / p * || K S1 ||_F^2
            expected_bias = 2*mu**2/p * np.sum((K @ S1)**2)
            # Variance = sigma^2 Tr(K); author estimates it by MC.
            exact_variance = SIGMA**2 * np.trace(K)
            noise_errors = cho_solve(fac, E1 + E2, check_finite=False)
            mc_variance = np.mean(np.sum(noise_errors**2, axis=0))

            records["direct"][r, j] = np.sum((beta_hps - beta2)**2)
            records["author"][r, j] = expected_bias + mc_variance
            records["finite_exact"][r, j] = expected_bias + exact_variance
            records["ols_direct"][r, j] = ols_direct
            records["ols_conditional"][r, j] = ols_conditional

    # Average over outer Monte Carlo repetitions; retain MC standard errors.
    rows = []
    for j, n1 in enumerate(n1_grid):
        row = {"n1": n1}
        for method, values in records.items():
            row[method] = float(np.mean(values[:, j]))
            row[method + "_mc_se"] = (float(np.std(values[:, j], ddof=1) / np.sqrt(reps))
                                       if reps > 1 else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# 4. Plotting and saving results
# ============================================================
def grid(max_n1, step):
    return list(range(0, max_n1 + 1, step))


def setting_name(fig, n2, p, lam, mu):
    if fig == "2a":
        return f"n2={n2}, p={p}"
    if fig in ("2b", "4a"):
        return f"lambda={lam:g}"
    return f"mu={mu:g}"


def plot_figure(fig, simulations, theories):
    """Theoretical lines and finite-sample MC points (plus all methods)."""
    fig_obj, ax = plt.subplots(figsize=(9, 5.8))
    for i, (label, curve) in enumerate(theories.groupby("setting", sort=False)):
        samples = simulations[simulations.setting == label].sort_values("n1")
        color = f"C{i}"
        curve = curve.sort_values("n1")
        ax.plot(curve.n1, curve.theory, color=color, lw=2.7,
                label=f"Theory: {label}", zorder=5)
        ax.scatter(samples.n1, samples.finite_exact, color=color,
                   marker="x", s=20, alpha=0.8, zorder=6)
        ax.scatter(samples.n1, samples.author, color=color,
                   marker="+", s=16, alpha=0.4, zorder=3)
        ax.scatter(samples.n1, samples.direct, color=color,
                   marker="o", s=9, alpha=0.25, zorder=2)

    ax.scatter([], [], marker="x", color="black", label="Finite exact")
    ax.scatter([], [], marker="+", color="black", label="Author bias + MC variance")
    ax.scatter([], [], marker="o", color="black", label="Direct MC")
    if fig == "4b":
        first = simulations.iloc[0]
        ols = SIGMA**2 * first.p / (first.n2 - first.p)
        ax.axhline(ols, color="gray", ls="--", lw=2, label="Target-only OLS")

    titles = {
        "2a": "Figure 2(a): varying sample size and dimension",
        "2b": "Figure 2(b): varying covariate shift",
        "4a": "Figure 4(a): covariance and model shifts",
        "4b": "Figure 4(b): varying model shift",
    }
    ax.set(title=titles[fig], xlabel="Source sample size $n_1$",
           ylabel="Target excess risk")
    ax.grid(ls=":", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig_obj.tight_layout()
    filename = OUTPUT_DIR / f"figure_{fig}"
    fig_obj.savefig(filename.with_suffix(".png"), dpi=200)
    fig_obj.savefig(filename.with_suffix(".pdf"))
    plt.close(fig_obj)


def run_figure(fig, reps, noise_trials, rng):
    cfg = FIGURES[fig]
    n1_grid = grid(cfg["max_n1"], cfg["sim_step"])
    theory_grid = grid(cfg["max_n1"], cfg["theory_step"])
    simulations, theories = [], []

    for n2, p, lam, mu in cfg["settings"]:
        label = setting_name(fig, n2, p, lam, mu)
        print(f"Figure {fig}: {label}, repetitions={reps}", flush=True)
        sim = simulate_one_setting(n1_grid, n2, p, lam, mu,
                                   reps, noise_trials, rng)
        sim["setting"] = label
        sim["figure"] = fig
        sim["n2"], sim["p"], sim["lambda"], sim["mu"] = n2, p, lam, mu
        sim["theory"] = [theoretical_risk(fig, n1, n2, p, lam, mu)
                         for n1 in sim.n1]
        sim["abs_error_direct"] = abs(sim.direct - sim.theory)
        sim["abs_error_author"] = abs(sim.author - sim.theory)
        sim["abs_error_finite_exact"] = abs(sim.finite_exact - sim.theory)
        simulations.append(sim)

        theories.append(pd.DataFrame({
            "setting": label, "figure": fig,
            "n1": theory_grid,
            "theory": [theoretical_risk(fig, n1, n2, p, lam, mu)
                       for n1 in theory_grid],
        }))

    simulations = pd.concat(simulations, ignore_index=True)
    theories = pd.concat(theories, ignore_index=True)
    simulations.to_csv(OUTPUT_DIR / f"figure_{fig}_simulation.csv", index=False)
    theories.to_csv(OUTPUT_DIR / f"figure_{fig}_theory.csv", index=False)
    plot_figure(fig, simulations, theories)
    return simulations


def check_theory():
    """Two simple checks against the earlier Figure 1 formula."""
    for n1 in (0, 100, 500, 1000):
        n2, p, mu = 300, 100, 0.3
        N = n1 + n2
        # If lambda=1, Theorem 10 reduces to Theorem 6.
        fig1 = (SIGMA**2 * p/(N-p)
                + 2*mu**2*(n1*n1*(N-p)+p*n1*n2)/(N*N*(N-p)))
        assert np.isclose(theoretical_risk("4b", n1, n2, p, 1, mu), fig1,
                          rtol=1e-6)
        # If mu=0, Theorem 10 reduces to Theorem 3.
        assert np.isclose(theoretical_risk("4b", n1, n2, p, 4, 0),
                          theoretical_risk("2b", n1, n2, p, 4, 0), rtol=1e-6)


def main():
    parser = argparse.ArgumentParser(description="Reproduce Figures 2 and 4")
    parser.add_argument("--figures", nargs="+", choices=list(FIGURES),
                        default=list(FIGURES), help="Figures to reproduce; default: all")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="Outer Monte Carlo repetitions (default: 40)")
    parser.add_argument("--noise-trials", type=int, default=DEFAULT_NOISE_TRIALS,
                        help="Noise repetitions for author-style variance (default: 20)")
    args = parser.parse_args()
    if args.reps < 1 or args.noise_trials < 1:
        parser.error("--reps and --noise-trials must be >= 1")

    check_theory()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    all_results = [run_figure(fig, args.reps, args.noise_trials, rng)
                   for fig in args.figures]
    full = pd.concat(all_results, ignore_index=True)
    full.to_csv(OUTPUT_DIR / "all_simulation_results.csv", index=False)
    columns = ["abs_error_direct", "abs_error_author", "abs_error_finite_exact"]
    summary = full.groupby(["figure", "setting"], sort=False)[columns].mean()
    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv")
    print("\nFinished. Results saved in:", OUTPUT_DIR)
    print(summary.round(5))


if __name__ == "__main__":
    main()
