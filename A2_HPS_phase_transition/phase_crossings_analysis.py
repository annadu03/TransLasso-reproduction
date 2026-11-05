#!/usr/bin/env python3
"""Figure 4(b): locate Theorem 10 HPS-vs-OLS crossings for mu=0.27.

The report uses n1 in (0, 6000]. This is a theory-only diagnostic;
Monte Carlo validation requires a separate simulation.
Run from any directory:
    python phase_crossings_analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from reproduce_figures_2_4 import figure4_theory, SIGMA

P, N2, LAM, MU = 100, 300, 4.0, 0.27
OLS = SIGMA**2 * P / (N2 - P)
OUT = Path(__file__).resolve().parent.parent

def components(n1):
    variance, bias = figure4_theory(float(n1), N2, P, LAM, MU)
    return variance, bias

def delta(n1):
    variance, bias = components(n1)
    return variance + bias - OLS

def main():
    ngrid = np.linspace(1, 6000, 6000)
    v, b = np.array([components(n) for n in ngrid]).T
    delta_values = v + b - OLS
    pairs = np.flatnonzero(delta_values[:-1] * delta_values[1:] < 0)
    crossings = [brentq(delta, ngrid[i], ngrid[i + 1]) for i in pairs]
    print('OLS risk:', OLS)
    print('Theorem-10 crossing points in (0,6000):', crossings)
    print('Theory only; do not interpret crossings as independently validated MC boundaries.')
    rows = []
    for n in [100, 200, 300, 600, 700, 1000, 6000]:
        var, bias = components(n)
        rows.append({'n1': n, 'variance': var, 'bias': bias,
                     'HPS_theory': var + bias, 'OLS_theory': OLS,
                     'delta': var + bias - OLS})
    outcsv = OUT / 'results' / 'mu027_theory_components.csv'
    outcsv.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows).to_csv(outcsv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.1))
    ax = axes[0]
    ax.plot(ngrid, v + b, label='HPS: Theorem 10', lw=2)
    ax.axhline(OLS, ls='--', color='gray', label='Target-only OLS')
    for c in crossings:
        ax.axvline(c, color='black', lw=0.9, alpha=0.5, ls=':')
    ax.set(xlabel='Source sample size n1', ylabel='Target excess risk',
           title='(a) HPS vs OLS, mu = 0.27')
    ax.legend(fontsize=8);ax.grid(alpha=0.15)
    ax = axes[1]
    ax.plot(ngrid, v, label='Variance', lw=2)
    ax.plot(ngrid, b, label='Bias', lw=2)
    ax.plot(ngrid, delta_values, label='HPS - OLS', lw=2, ls='--')
    ax.axhline(0, color='gray', lw=0.8)
    for c in crossings:
        ax.axvline(c, color='black', lw=0.9, alpha=0.5, ls=':')
    ax.set(xlabel='Source sample size n1', ylabel='Theoretical component / risk difference',
           title='(b) Bias-variance decomposition')
    ax.legend(fontsize=8);ax.grid(alpha=0.15)
    fig.suptitle('Two predicted transfer-effect sign changes within n1 <= 6000')
    fig.tight_layout()
    outfig= OUT / 'figures' / 'mu027_crossings.png'
    outfig.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outfig, dpi=200, bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    main()
