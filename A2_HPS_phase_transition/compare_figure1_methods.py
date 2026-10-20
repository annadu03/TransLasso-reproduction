"""Compare two empirical Figure 1 simulations with Theorem 6 asymptotics.

Paper: Yang et al. (2025), Precise High-Dimensional Asymptotics for
Quantifying Heterogeneous Transfers, Section 4 / Figure 1.

Run: python compare_figure1_methods.py
Optional: python compare_figure1_methods.py --reps 100 --step 20
Extra: python compare_figure1_methods.py --mu-list 0.2 0.25 0.3 0.35 0.4 0.45
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


def theoretical_risk(n1, n2, p, sigma, mu):
    """Theorem 6 leading term under isotropic random effects."""
    n = n1 + n2
    variance = sigma**2 * p / (n - p)
    bias = 2 * mu**2 * (n1**2 * (n - p) + p * n1 * n2) / (n**2 * (n - p))
    return bias + variance


def run_one_mu(mu, n1_grid, p, n2, sigma, reps, noise_trials, rng):
    """Average over independently sampled Gaussian designs and model shifts."""
    names = ("direct", "author", "finite_exact", "ols_direct", "ols_conditional")
    values = {name: np.zeros(len(n1_grid)) for name in names}
    eye = np.eye(p)

    for _ in range(reps):
        # Keep target data fixed while varying source sample size within this replication.
        X2 = rng.normal(size=(n2, p))
        eps2 = rng.normal(scale=sigma, size=n2)
        X1_full = rng.normal(size=(int(max(n1_grid)), p))
        eps1_full = rng.normal(scale=sigma, size=len(X1_full))

        # Random-effects model: beta_i = beta0 + gamma_i, gamma_i ~ N(0, mu^2/p I).
        beta0 = np.zeros(p)  # Shared component cancels out of excess risk.
        beta1 = beta0 + rng.normal(scale=mu / np.sqrt(p), size=p)
        beta2 = beta0 + rng.normal(scale=mu / np.sqrt(p), size=p)

        S2 = X2.T @ X2
        y2 = X2 @ beta2 + eps2
        beta_ols = np.linalg.solve(S2, X2.T @ y2)
        ols_direct = np.sum((beta_ols - beta2) ** 2)
        ols_conditional = sigma**2 * np.trace(np.linalg.solve(S2, eye))

        for j, n1 in enumerate(n1_grid):
            X1 = X1_full[:n1]
            eps1 = eps1_full[:n1]
            S1 = X1.T @ X1
            K = np.linalg.solve(S1 + S2, eye)

            # Method A (ours): explicitly generate TWO distinct true models.
            y1 = X1 @ beta1 + eps1
            beta_hps = K @ (X1.T @ y1 + X2.T @ y2)
            direct_risk = np.sum((beta_hps - beta2) ** 2)

            # Method B (authors' notebook): exact conditional bias + MC variance.
            # The variance experiment uses a COMMON true beta in both tasks;
            # after subtracting that beta, only Gaussian noise remains.
            mc_eps1 = rng.normal(scale=sigma, size=(n1, noise_trials))
            mc_eps2 = rng.normal(scale=sigma, size=(n2, noise_trials))
            mc_errors = K @ (X1.T @ mc_eps1 + X2.T @ mc_eps2)
            mc_variance = np.mean(np.sum(mc_errors**2, axis=0))

            A = K @ S1
            expected_bias = (2 * mu**2 / p) * np.sum(A**2)
            author_risk = expected_bias + mc_variance

            # Finite-design conditional expectation (not Theorem 6):
            # exact in expectation over noise and task-specific coefficients.
            finite_exact = expected_bias + sigma**2 * np.trace(K)

            values["direct"][j] += direct_risk
            values["author"][j] += author_risk
            values["finite_exact"][j] += finite_exact
            values["ols_direct"][j] += ols_direct
            values["ols_conditional"][j] += ols_conditional

    return {key: arr / reps for key, arr in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=40, help="Monte Carlo dataset replications")
    parser.add_argument("--noise-trials", type=int, default=20,
                        help="Noise trials for the author's empirical variance")
    parser.add_argument("--step", type=int, default=50, help="Increment of source sample size")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--mu-list", type=float, nargs="+",
        default=[0.15, 0.20, 0.24, 0.26, 0.28, 0.30,
                 0.32, 0.35, 0.38, 0.40, 0.43, 0.45],
        help="Model-shift values for the simulation plots",
    )
    args = parser.parse_args()
    if args.reps < 1 or args.noise_trials < 1 or args.step < 1:
        parser.error("reps, noise-trials and step must be positive")

    # Figure 1 experimental settings.
    p, n2, sigma = 100, 300, 0.5
    mu_list = sorted(set(args.mu_list))
    if any(mu < 0 for mu in mu_list):
        parser.error("mu must be nonnegative")
    n1_grid = np.unique(np.append(np.arange(0, 1001, args.step), 1000))
    rng = np.random.default_rng(args.seed)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dynamic number of panels: unlike a fixed 2x2, all requested mu values are shown.
    ncols = min(4, len(mu_list))
    nrows = int(np.ceil(len(mu_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.4*nrows),
                             sharex=True, sharey=True, squeeze=False)
    all_rows = []
    ols_theory = sigma**2 * p / (n2 - p)

    for ax, mu in zip(axes.flat, mu_list):
        result = run_one_mu(mu, n1_grid, p, n2, sigma,
                            args.reps, args.noise_trials, rng)
        theory = np.array([theoretical_risk(n1, n2, p, sigma, mu)
                           for n1 in n1_grid])

        for j, n1 in enumerate(n1_grid):
            all_rows.append({
                "mu": mu, "n1": int(n1), "n2": n2, "p": p,
                "direct_risk": result["direct"][j],
                "author_bias_variance_risk": result["author"][j],
                "finite_conditional_risk": result["finite_exact"][j],
                "theorem6_risk": theory[j],
                "ols_direct_risk": result["ols_direct"][j],
                "ols_conditional_risk": result["ols_conditional"][j],
                "ols_theory_risk": ols_theory,
            })

        # Draw simulations first with transparency, so the blue theory line stays visible.
        ax.plot(n1_grid, result["author"], "o-", ms=2.5, alpha=0.55,
                linewidth=1.0, label="Author: bias + MC variance")
        ax.plot(n1_grid, result["direct"], "s--", ms=2.5, alpha=0.55,
                linewidth=1.0, label="Direct: two different betas")
        ax.axhline(ols_theory, color="gray", linestyle=":", linewidth=1.8,
                   label="OLS theory")
        ax.plot(n1_grid, theory, color="blue", linewidth=2.7, zorder=10,
                label="Theorem 6")
        ax.set_title(f"Model shift: mu = {mu:.2f}")
        ax.grid(alpha=0.25)

    # Hide unused cells if the mu count does not fill a rectangular grid.
    for ax in axes.flat[len(mu_list):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Source sample size n1")
    for ax in axes[:, 0]:
        ax.set_ylabel("Target excess risk")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"Figure 1: two simulations vs high-dimensional theory (reps={args.reps})")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(out_dir / "figure1_method_comparison.png", dpi=180)
    plt.close(fig)

    # Phase diagram based ONLY on Theorem 6 (not on empirical simulations).
    # Red: risk difference > 0 (negative transfer); blue: < 0 (positive transfer).
    mu_fine = np.linspace(min(mu_list), max(mu_list), 241)
    n1_fine = np.linspace(1, 1000, 300)  # n1=0 is always exactly tied with OLS
    NN, MM = np.meshgrid(n1_fine, mu_fine)
    risk_diff = theoretical_risk(NN, n2, p, sigma, MM) - ols_theory

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    limit = float(np.max(np.abs(risk_diff)))
    im = ax2.pcolormesh(
        NN, MM, risk_diff, shading="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    ax2.contour(NN, MM, risk_diff, levels=[0], colors="black", linewidths=2)
    cbar = fig2.colorbar(im, ax=ax2)
    cbar.set_label("HPS risk - OLS risk")
    mu_low = np.sqrt(sigma**2 * p / (2 * (n2 - p)))
    mu_high = np.sqrt(sigma**2 * n2 / (2 * (n2 - p)))
    if mu_fine[0] <= mu_low <= mu_fine[-1]:
        ax2.axhline(mu_low, color="black", alpha=0.55, linestyle="--", linewidth=0.9)
    if mu_fine[0] <= mu_high <= mu_fine[-1]:
        ax2.axhline(mu_high, color="black", alpha=0.55, linestyle="--", linewidth=0.9)
    ax2.set_xlabel("Source sample size n1")
    ax2.set_ylabel("Model shift mu")
    ax2.set_title("Theorem 6 phase diagram: blue = positive, red = negative transfer")
    fig2.tight_layout()
    fig2.savefig(out_dir / "phase_diagram_theory.png", dpi=200)
    plt.close(fig2)

    csv_file = out_dir / "figure1_method_comparison.csv"
    pd.DataFrame(all_rows).to_csv(csv_file, index=False)
    print(f"Saved: {out_dir / 'figure1_method_comparison.png'}")
    print(f"Saved: {csv_file}")
    print(f"Saved: {out_dir / 'phase_diagram_theory.png'}")
    print("OLS theoretical excess risk:", round(ols_theory, 6))


if __name__ == "__main__":
    main()
