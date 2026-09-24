"""Frozen alignment diagnostic; no training, labels, or simulator access.

Loop from mte_attribution_resolution_2026_09_12/mamujoco_probe_audit.py.
Caller supplies the archived row selection and canonical training scale.
"""
import numpy as np
from evaluation.probes import reader, transform

def score(latents, actual, reference, masks, mobius, canonical_std, train_rows, dev_rows):
    tr=np.asarray(train_rows);dv=np.asarray(dev_rows)
    assert tr.ndim==dv.ndim==2 and tr.shape[1]==dv.shape[1]==2
    assert tr[:,1].min()>0 and dv[:,1].min()>0
    def target(ix):
        e,t=ix.T
        return transform((actual[e,t].astype(float)-reference[e,t].astype(float))/canonical_std,np.asarray(mobius,dtype=float))
    yt,yv=target(tr),target(dv);rows=[]
    for shift in [-1,0,1]:
        zt=latents[tr[:,0],tr[:,1]+shift].astype(float)
        zv=latents[dv[:,0],dv[:,1]+shift].astype(float)
        pred=reader(zt,yt.reshape(len(yt),-1),zv).reshape(yv.shape)
        for h in range(3):
            for order in [0,1,2,3]:
                sel=masks.sum(1)==order
                mse=float(np.mean((pred[:,h,sel]-yv[:,h,sel])**2))
                den=float(np.mean((yv[:,h,sel]-yt[:,h,sel].mean(0))**2))
                rows.append(dict(shift=shift,horizon=h+1,order=order,mse=mse,baseline_mse=den,skill=1-mse/den if den>1e-12 else None))
    return rows
