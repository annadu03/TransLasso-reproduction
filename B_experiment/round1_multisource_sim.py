#!/usr/bin/env python3
"""Round 1: multi-source HPS simulations under model/covariate/joint shifts.

Uses exact noise-conditional target excess risk (not a test-set estimate).
Install: pip install numpy pandas scipy matplotlib
Run: python round1_multisource_sim.py --reps 200 --outdir outputs/round1
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def trace_normalized_scales(p, lam, pattern):
    """Both source covariances have trace p and the same multiset of eigenvalues.

    'aligned'  means Sigma1 == Sigma2; 'complement' swaps the two halves.
    """
    if p % 2:
        raise ValueError('p must be even')
    a = 2.0 * lam / (lam + 1.0 / lam)
    b = 2.0 / lam / (lam + 1.0 / lam)
    s1 = np.r_[np.full(p//2, a), np.full(p//2, b)]
    s2 = s1.copy() if pattern == 'aligned' else s1[::-1].copy()
    return np.sqrt(s1), np.sqrt(s2)


def v_and_solutions(H, rhs):
    c = cho_factor(H, lower=True, overwrite_a=False, check_finite=False)
    z = cho_solve(c, rhs, check_finite=False)
    inv_h = cho_solve(c, np.eye(H.shape[0]), check_finite=False)
    return float(np.trace(inv_h)), z


def experiment_specs(lambdas):
    specs = [('none', 1., 'aligned'), ('model', 1., 'aligned')]
    for lam in lambdas:
        specs += [('cov_aligned', lam, 'aligned'),
                  ('cov_complement', lam, 'complement'),
                  ('joint_aligned', lam, 'aligned'),
                  ('joint_complement', lam, 'complement')]
    return specs


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p, n0, n1, n2 = args.p, args.n0, args.n1, args.n2
    sig2 = args.sigma**2
    u = np.zeros(p); u[0] = 1.0
    v = np.zeros(p); v[p//2] = 1.0
    records = []
    # Identical base standard-normal draws across covariance scenarios,
    # making aligned/complement comparisons paired by replicate.
    specs = experiment_specs(args.lambdas)
    for rep in range(args.reps):
        rng = np.random.default_rng(args.seed + rep)
        z0 = rng.standard_normal((n0,p))
        z1 = rng.standard_normal((n1,p))
        z2 = rng.standard_normal((n2,p))
        A0 = z0.T @ z0
        ch0 = cho_factor(A0, lower=True, check_finite=False)
        v0 = sig2 * float(np.trace(cho_solve(ch0, np.eye(p), check_finite=False)))
        for mode, lam, pattern in specs:
            scl1, scl2 = trace_normalized_scales(p, lam, pattern)
            x1 = z1 * scl1
            x2 = z2 * scl2
            A1, A2 = x1.T @ x1, x2.T @ x2
            # The three alternatives evaluated under the same draw of X.
            tr1, z1u = v_and_solutions(A0+A1, A1@u)
            tr2, zs2 = v_and_solutions(A0+A2, np.column_stack((A2@u, A2@v)))
            tr12, zsboth = v_and_solutions(A0+A1+A2,
                                           np.column_stack((A1@u, A2@u, A2@v)))
            vv1, vv2, vv12 = sig2*tr1, sig2*tr2, sig2*tr12
            mus = [0.] if mode in ('none', 'cov_aligned', 'cov_complement') else args.mus
            rhos = [-1.] if mode in ('none', 'cov_aligned', 'cov_complement') else args.rhos
            for mu in mus:
                for rho in rhos:
                    orth = np.sqrt(max(0., 1. - rho*rho))
                    shift2 = rho*zs2[:, 0] + orth*zs2[:, 1]
                    joint_shift = zsboth[:, 0] + rho*zsboth[:, 1] + orth*zsboth[:, 2]
                    b1 = mu*mu*float(z1u@z1u)
                    b2 = mu*mu*float(shift2@shift2)
                    b12 = mu*mu*float(joint_shift@joint_shift)
                    r1, r2, r12 = vv1+b1, vv2+b2, vv12+b12
                    g1,g2,g12 = v0-r1,v0-r2,v0-r12
                    records.append({
                        'rep':rep, 'mode':mode,'lambda':lam,'cov_pattern':pattern,
                        'mu':mu,'rho':rho,
                        'R_target':v0,'R_s1':r1,'R_s2':r2,'R_joint':r12,
                        'B_s1':b1,'B_s2':b2,'B_joint':b12,
                        'V_target':v0,'V_s1':vv1,'V_s2':vv2,'V_joint':vv12,
                        'G_s1':g1,'G_s2':g2,'G_joint':g12,
                        'Gamma':g12-g1-g2,
                        'reversal':int(g1<0 and g2<0 and g12>0)
                    })
        if (rep+1)%50==0 or (rep+1)==args.reps:
            print(f'Finished {rep+1}/{args.reps} repetitions', flush=True)
    raw = pd.DataFrame.from_records(records)
    raw.to_csv(outdir/'round1_replicates.csv', index=False)
    groups = ['mode','lambda','cov_pattern','mu','rho']
    metrics=['R_target','R_s1','R_s2','R_joint','B_s1','B_s2','B_joint',
             'V_target','V_s1','V_s2','V_joint','G_s1','G_s2','G_joint','Gamma','reversal']
    agg=raw.groupby(groups, as_index=False)[metrics].agg(['mean','std','count'])
    # flatten hierarchical groupby columns
    agg.columns=['_'.join(x).rstrip('_') for x in agg.columns.to_flat_index()]
    for c in ['G_s1','G_s2','G_joint','Gamma']:
        agg[c+'_mc95low'] = agg[c+'_mean'] - 1.96*agg[c+'_std']/np.sqrt(agg[c+'_count'])
        agg[c+'_mc95high'] = agg[c+'_mean'] + 1.96*agg[c+'_std']/np.sqrt(agg[c+'_count'])
    agg.to_csv(outdir/'round1_summary.csv',index=False)

    # Main risk-vs-model-shift plot for anti-aligned shifts
    fig,ax=plt.subplots(figsize=(8,5))
    dd=agg[(agg['mode']=='model')&(agg['rho']==-1.)].sort_values('mu')
    for label,col in [('Target only','R_target_mean'),('Target + source 1','R_s1_mean'),
                      ('Target + source 2','R_s2_mean'),('Target + both','R_joint_mean')]:
        ax.plot(dd['mu'],dd[col],marker='o',label=label)
    ax.set(xlabel='Model shift magnitude mu',ylabel='Target excess risk (expected over noise)',
           title='Model shift only: opposite source biases (rho=-1)')
    ax.legend();fig.tight_layout();fig.savefig(outdir/'fig_model_shift_rho_minus1.png',dpi=180)
    plt.close(fig)

    # Covariate-shift-only: aligned versus complementary covariance orientation
    fig,ax=plt.subplots(figsize=(8,5))
    dd=agg[agg['mode'].isin(['cov_aligned','cov_complement'])]
    for mode,label in [('cov_aligned','Aligned source spectra'),
                       ('cov_complement','Complementary source spectra')]:
        q=dd[dd['mode']==mode].sort_values('lambda')
        ax.plot(q['lambda'],q['R_joint_mean'],marker='o',label=label)
    ax.axhline(raw['R_target'].mean(),ls='--',label='Target only')
    ax.set(xlabel='Eigenvalue contrast lambda (trace normalized)',ylabel='Target excess risk',
           title='Covariate shift only: covariance complementarity')
    ax.legend();fig.tight_layout();fig.savefig(outdir/'fig_covariance_only.png',dpi=180)
    plt.close(fig)

    # Joint shift, complementary covariance spectra: compare effective offset for two mu values
    fig,ax=plt.subplots(figsize=(8,5))
    for lam in args.lambdas:
        for rho in [-1.,1.]:
            q=agg[(agg['mode']=='joint_complement')&(agg['lambda']==lam)&(agg['rho']==rho)].sort_values('mu')
            ax.plot(q['mu'],q['G_joint_mean'],marker='o',label=f'lambda={lam:g}, rho={rho:g}')
    ax.axhline(0,ls='--')
    ax.set(xlabel='Model shift magnitude mu',ylabel='Gain from using both sources',
           title='Joint covariate + model shifts')
    ax.legend();fig.tight_layout();fig.savefig(outdir/'fig_joint_shift.png',dpi=180)
    plt.close(fig)
    print('Saved:', outdir.resolve())
    print('Files: round1_replicates.csv, round1_summary.csv, 3 figures')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p',type=int,default=100)
    parser.add_argument('--n0',type=int,default=300)
    parser.add_argument('--n1',type=int,default=150)
    parser.add_argument('--n2',type=int,default=150)
    parser.add_argument('--sigma',type=float,default=.5)
    parser.add_argument('--reps',type=int,default=200)
    parser.add_argument('--seed',type=int,default=20261118)
    parser.add_argument('--lambdas',type=float,nargs='+',default=[2.,4.])
    parser.add_argument('--mus',type=float,nargs='+',default=[0.,.3,.6,.8,1.,1.2])
    parser.add_argument('--rhos',type=float,nargs='+',default=[-1.,0.,1.])
    parser.add_argument('--outdir',type=str,default='outputs/round1_multisource')
    args=parser.parse_args()
    if args.p<=0 or args.p%2 or args.n0<=args.p:
        parser.error('p must be positive/even and n0 > p')
    if args.reps<2 or any(abs(r)>1 for r in args.rhos) or any(l<1 for l in args.lambdas):
        parser.error('reps >=2, -1<=rho<=1, lambda>=1 required')
    run(args)

if __name__=='__main__':
    main()
