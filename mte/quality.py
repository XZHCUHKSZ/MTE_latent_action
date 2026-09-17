import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from closed_loop_lam_v1 import common as C

# Final implementation source: mamujoco_smalln_quality_recovery_2026_09_12/quality.py:3

def evaluate(x,m,n,a,b):
    delta=(x[:n,1:]-x[:n,:-1])[m[:n]].astype(np.float64)
    constant=np.ptp(delta,axis=0)==0;active=~constant;scale=delta.std(0).clip(1e-6)
    drift=delta.mean(0);tolerance=1e-4*(1+np.abs(x[:n,:-1][m[:n]]).max(0))
    finite=bool(np.isfinite(a).all() and np.isfinite(b).all());rows=[];t=np.arange(m.shape[1])
    for h in [1,2,3]:
        valid=m[:,np.minimum(t+h-1,m.shape[1]-1)]&(t+h<=m.shape[1])[None]
        target=x[:,np.minimum(t+h,x.shape[1]-1)].astype(np.float64)
        base=x[:,:m.shape[1]].astype(np.float64);err=(a[n:,:,h-1,0].astype(np.float64)-target[n:])[valid[n:]]
        persistence=(base[n:]-target[n:])[valid[n:]]
        raw=float(np.square(err).mean());pr=float(np.square(persistence).mean())
        standardized=float(np.square(err[:,active]/scale[active]).mean()) if active.any() else 0.
        maxratio=0.
        if constant.any():
            for endpoints in [a,b]:
                predicted=endpoints[:,:,h-1][...,constant]
                expected=base[...,constant]+h*drift[constant]
                residual=(predicted-expected[:,:,None,:])[valid]
                maxratio=max(maxratio,float(np.max(np.abs(residual)/tolerance[constant])))
        passed=finite and standardized<10 and raw<=10*pr+1e-10 and maxratio<=1
        rows.append(dict(h=h,passed=passed,variable_coordinate_standardized_mse=standardized,
            original_all_coordinate_standardized_mse=float(np.square(err/scale).mean()),
            all_coordinate_raw_mse=raw,persistence_raw_mse=pr,constant_prediction_drift_tolerance_ratio=maxratio,
            per_column_raw_mse=np.square(err).mean(0).tolist(),
            constant_dev_changed_counts=np.count_nonzero(persistence[:,constant],axis=0).tolist()))
    return dict(passed=all(r['passed'] for r in rows),all_branch_values_finite=finite,
        training_constant_increment_columns=np.flatnonzero(constant).tolist(),horizons=rows)
