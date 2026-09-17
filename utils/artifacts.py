"""Explicit artifact routing, separated from scientific functions and job history.

Pass a suite with train, dev, endpoints and (only for parity audits) original_bridge.
No automatic fallback to intermediate endpoints is allowed.
"""
from pathlib import Path
import numpy as np

def oldpath(s,stage):
    if stage!='bridge':raise ValueError('Only immutable original bridge parity inputs are supported')
    return Path(s['original_bridge'])

old=oldpath

def endpoints(s):
    p=Path(s['endpoints'])
    if s['env']=='mamujoco' and s.get('endpoint_version')!='constant_support_repaired':
        raise ValueError('Final MaMuJoCo experiments require constant-support-repaired endpoints')
    return p

def data(s):
    observations=[];masks=[]
    for split in ('train','dev'):
        with np.load(s[split],allow_pickle=False) as f:
            expected={'positions'} if s['env']=='mpe' else {'observations','valid_mask'}
            if set(f.files)!=expected:raise ValueError('Unlabelled export has unexpected fields')
            x=f['positions' if s['env']=='mpe' else 'observations'].astype(np.float32)
            mask=np.ones((len(x),x.shape[1]-1),bool) if s['env']=='mpe' else f['valid_mask'].astype(bool)
        observations.append(x);masks.append(mask)
    n=len(observations[0])
    if n!=s['n']:raise ValueError('Protocol training count differs from export')
    return np.concatenate(observations),np.concatenate(masks),n

def rich(s):
    from training.representation_mpe import endpoint_view
    from mte.entity_view import entities
    x,m,n=data(s)
    with np.load(endpoints(s),allow_pickle=False) as f:
        a,b,cm=f['actual'],f['reference'],f['masks']
    if s['env']=='mpe':
        view=endpoint_view(x,a,b,cm)
        return x,m,n,view['cf_horizon_entity_root'],view['cf_horizon_entity_ego_null'],cm,view['mif_entity_adjacency'],m[:,1:23]
    return x,m,n,entities(a),entities(b),cm,np.ones((4,4),np.float32),m
