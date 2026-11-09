#!/usr/bin/env python3
"""Rebuild Figures 7–9 directly from the archived Bratu predictions.

This is an independently reconstructed plotting implementation: original PDF
assets were present but the plotting source for Figures 7–9 was not supplied.
The mathematical curves and run identity are verified against the manuscript.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

LAMBDAS=(1.,2.,3.,3.4)
METHODS=('lra','relobralo','ddqn','fixed_selected')
NAMES={'lra':'LRA','relobralo':'ReLoBRaLo','ddqn':'DDQN','fixed_selected':'Fixed (1, 3)'}
COLORS={'lra':'#1f77b4','relobralo':'#ff7f0e','ddqn':'#2ca02c','fixed_selected':'#d62728'}
STYLES={'lra':'--','relobralo':':','ddqn':'-.','fixed_selected':'--'}


def lambda_at_a(a):
    a=np.asarray(a)
    return 2*a*a/(np.cosh(a/2)**2)


def profile(a,x):
    return 2*np.log(np.cosh(a/2)/np.cosh(a*(np.asarray(x)-0.5)))


def roots(lam):
    acrit=brentq(lambda a: a*np.tanh(a/2)-2,2,3,xtol=1e-14)
    lamcrit=float(lambda_at_a(acrit))
    if not (0 < lam < lamcrit):
        raise ValueError('Subcritical Bratu parameter required')
    low=brentq(lambda a:float(lambda_at_a(a)-lam),1e-10,acrit,xtol=1e-13)
    high=brentq(lambda a:float(lambda_at_a(a)-lam),acrit,20,xtol=1e-13)
    return low,high,acrit,lamcrit


def selected_file(results,method,lam,seed):
    path=results/'runs'/f"{method}_lambda_{str(lam).replace('.', 'p')}_seed_{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f'Archived selected run missing: {path}')
    z=np.load(path,allow_pickle=False)
    for key in ('x','y_pred','y_ref'):
        if key not in z.files:
            raise KeyError(f'{path} is missing {key}')
    return z


def save(fig,output,name):
    fig.tight_layout()
    fig.savefig(output/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(output/f'{name}.png',dpi=220,bbox_inches='tight')
    plt.close(fig)


def draw_figure7(output):
    alo,ahi,acrit,lamcrit=roots(1.)
    a=np.linspace(0.0001,ahi,1250)
    lams=lambda_at_a(a)
    amps=2*np.log(np.cosh(a/2))
    left=a<=acrit
    fig,ax=plt.subplots(figsize=(7.0,4.4))
    ax.plot(lams[left],amps[left],color='#1f77b4',lw=1.8,label='Lower branch')
    ax.plot(lams[~left],amps[~left],color='#ff7f0e',lw=1.6,ls='--',label='Upper branch')
    for lam in LAMBDAS:
        low,high,_,_=roots(lam)
        ax.scatter([lam,lam],[profile(low,0.5),profile(high,0.5)],s=14,color='#2ca02c',zorder=4)
    ax.scatter([lamcrit],[profile(acrit,0.5)],s=110,marker='*',color='#9467bd',
               label=r'Critical point ($\lambda_c\approx3.51383$)',zorder=5)
    for lam in (3.,3.4):
        ax.text(lam+0.03,0.08,f'{lam:g}',rotation=90,fontsize=8,color='#444444')
    ax.set_xlim(0.92,3.67)
    ax.set_xlabel(r'Bratu parameter $\lambda$')
    ax.set_ylabel(r'Midpoint amplitude $u(1/2)$')
    ax.grid(True,alpha=0.22)
    ax.legend(frameon=False,loc='upper right',fontsize=8)
    save(fig,output,'Fig7')
    return lamcrit


def draw_figure8(results,output):
    fig,axes=plt.subplots(2,2,figsize=(8.6,5.6),sharex=True)
    legend={}
    for ax,lam in zip(axes.flat,LAMBDAS):
        z=selected_file(results,'lra',lam,123)
        x=z['x'].ravel(); low,_,_,_=roots(lam)
        y_ref=profile(low,x)
        ax.plot(x,y_ref,'-',lw=1.5,color='#333333',label='Reference: lower branch')
        for m in METHODS:
            zm=selected_file(results,m,lam,123)
            xx=zm['x'].ravel(); yy=zm['y_pred'].ravel()
            if not np.allclose(x,xx,atol=1e-14):
                raise ValueError(f'Incompatible test grids at lambda={lam}: {m}')
            ax.plot(xx,yy,STYLES[m],lw=1.15,color=COLORS[m],label=NAMES[m],alpha=.95)
        ax.set_title(r'$\lambda='+str(lam).rstrip('0').rstrip('.')+'$',fontsize=10)
        ax.grid(True,alpha=.23)
        if ax in axes[:,0]:ax.set_ylabel(r'$u(x)$')
        if ax in axes[-1,:]:ax.set_xlabel(r'$x$')
    h,l=axes.flat[0].get_legend_handles_labels()
    fig.legend(h,l,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.02),fontsize=8,frameon=False)
    fig.subplots_adjust(bottom=.17,wspace=.18,hspace=.30)
    # Unlike other figures, legend outside subplots needs manual margin.
    fig.savefig(output/'Fig8.pdf',bbox_inches='tight')
    fig.savefig(output/'Fig8.png',dpi=220,bbox_inches='tight')
    plt.close(fig)


def draw_figure9(results,output):
    lam=3.4; seed=456
    low,high,_,_=roots(lam)
    dz=selected_file(results,'ddqn',lam,seed); lz=selected_file(results,'lra',lam,seed)
    x=dz['x'].ravel(); yd=dz['y_pred'].ravel(); yl=lz['y_pred'].ravel()
    true_low=profile(low,x);true_high=profile(high,x)
    rms_low=float(np.sqrt(np.mean((yd-true_low)**2)))
    rms_upper=float(np.sqrt(np.mean((yd-true_high)**2)))
    if not (abs(rms_low-.0603)<.0001 and abs(rms_upper-.4789)<.0001):
        raise ValueError(f'Branch diagnostic mismatch: {rms_low:.5f}, {rms_upper:.5f}')
    fig,ax=plt.subplots(figsize=(7.0,4.3))
    ax.plot(x,true_low,color='#1f77b4',lw=1.8,label='Lower branch')
    ax.plot(x,true_high,color='#ff7f0e',ls=':',lw=1.6,label='Upper branch')
    ax.plot(x,yd,color='#2ca02c',ls='--',lw=1.4,label='DDQN, selected')
    ax.plot(x,yl,color='#d62728',ls='-.',lw=1.4,label='LRA, selected')
    ax.set_title(r'$\lambda=3.4$, outer seed 456',fontsize=10)
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$u(x)$')
    ax.grid(True,alpha=.23);ax.legend(frameon=False,loc='upper right',fontsize=8)
    save(fig,output,'Fig9')
    return rms_low,rms_upper


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,required=True,help='Path to results directory (runs/ inside)')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    critical=draw_figure7(args.out)
    draw_figure8(args.results,args.out)
    lo,hi=draw_figure9(args.results,args.out)
    print(f'Figures 7–9 rebuilt. critical lambda={critical:.12f}; Fig9 RMS: lower={lo:.6f}, upper={hi:.6f}')

if __name__=='__main__':main()
